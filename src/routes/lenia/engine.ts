import * as tf from '@tensorflow/tfjs';

const [h, w] = [16,16]
const bumps = 3;
const channels = 4;

const mu = tf.variable(tf.randomUniform([channels]))
const sigma = tf.variable(tf.randomUniform([channels]).mul(.1))

const alphas = tf.variable(tf.randomUniform([channels, bumps]).mul(4.))
const betas = tf.variable(tf.randomUniform([channels, bumps]))
const weights = tf.variable(tf.randomUniform([channels, bumps]))

const linear = tf.variable(tf.randomUniform([channels, channels], -1/Math.sqrt(channels), 1/Math.sqrt(channels)))

const dt = tf.scalar(.3);
const eps = tf.scalar(4e-8)

const filter = tf.tensor([1,4,4,1])

export const smooth = (x:tf.Tensor4D) => {
    const kernel:tf.Tensor2D = filter.reshape([1,4]).mul(filter.reshape([4,1]))
    const nkernel:tf.Tensor4D = kernel.div(kernel.sum()).reshape([4,4,1,1]).tile([1,1,4,1])
    return tf.depthwiseConv2d(x, nkernel, [1,1], "same")
}

export const convert = (x:tf.Tensor4D) => x.mean([-1,0]).mul(255).clipByValue(0,255).cast("int32")

const growth = (x:tf.Tensor) => {
    const top = x.sub(mu).square()
    const bottom = sigma.square().mul(2)
    
    const g = tf.exp(top.div(bottom).mul(-1))

    return g.mul(2).sub(1)

}

const radius = () => {
    const yi = tf.linspace(-h/2, h/2, h)
    const xi = tf.linspace(-w/2, w/2, w)
    const [yy, xx] = tf.meshgrid(yi, xi)
    
    return tf.sqrt(yy.square().add(xx.square()))
}


const constructKernel = (alphas:tf.Tensor, betas:tf.Tensor, weights:tf.Tensor): tf.Tensor4D => {
    
    const [_alphas, _betas, _weights] = [alphas, betas, weights].map(t => t.reshape([1,1,bumps,channels]))

    const r = radius().reshape([h, w, 1, 1]).tile([1,1,bumps,channels])
    const rd = _betas.mul(r.max([0,1], true))
    const radii = r.div(rd).clipByValue(0,1)    


    const dshell = _alphas.mul(radii).mul(radii.sub(1)).mul(-1)
    const shell = _alphas.div(dshell.add(eps))
    const kernel = tf.exp(_alphas.sub(shell)).mul(_weights).sum([-2])
    const dkernel = kernel.mean([0,1], true)

    return kernel.div(dkernel).reshape([h, w, channels, 1])
}

const normalise = (x:tf.Tensor, axis?:number[]) => {
    const mean = x.mean(axis, true)
    const variance = mean.squaredDifference(x).mean(axis, true)

    return x.sub(mean).div(variance.sqrt())
}

export const lenia = (world:tf.Tensor4D) => {

    const kernel = constructKernel(alphas, betas, weights)
    const acts = tf.depthwiseConv2d(world, kernel, [1,1], "same")
    const update = growth(acts.matMul(linear)).mul(dt)

    const noise = tf.randomUniform(world.shape)
    const nextWorld = world.add(update).add(noise)
    return normalise(nextWorld, [1,2])
}

// const [height, width] = [256, 256]
// const world:tf.Tensor4D = tf.randomUniform([1, height, width, channels])
// const nextWorld = lenia(world)