import * as tf from '@tensorflow/tfjs';

const seed = 42;
const height = 32;
const width = 32;
const bumps = 3;
const channels = 4;

const uniform = tf.initializers.randomUniform({seed})
const luniform = tf.initializers.leCunUniform({seed})

const mu = tf.variable(uniform.apply([channels]))
const sigma = tf.variable(uniform.apply([channels]).mul(.1))

const alphas = tf.variable(uniform.apply([channels, bumps]).mul(4.))
const betas = tf.variable(uniform.apply([channels, bumps]))
const weights = tf.variable(uniform.apply([channels, bumps]))
const linear = tf.variable(luniform.apply([channels, channels]))

const dt = tf.scalar(.3);
const eps = tf.scalar(4e-12)

const growth = (x:tf.Tensor) => {
    const top = x.sub(mu).square()
    const bottom = sigma.square().mul(2)
    
    const g = tf.exp(-top.div(bottom))

    return g.mul(2).sub(1)

}

const radius = () => {
    const yi = tf.linspace(-height/2, height/2, height)
    const xi = tf.linspace(-width/2, width/2, width)
    const [yy, xx] = tf.meshgrid(yi, xi)
    
    return tf.sqrt(yy.square().add(xx.square()))
}


const constructKernel = (alphas:tf.Tensor, betas:tf.Tensor, weights:tf.Tensor) => {
    
    const [_alphas, _betas, _weights] = [alphas, betas, weights].map(t => t.reshape([1,1,bumps,channels]))

    const r = radius().reshape([height, width, 1, 1]).tile([1,1,bumps,channels])
    const rd = _betas.mul(r.max([0,1], true))
    const radii = r.div(rd).clipByValue(0,1)
    

    const dshell = _alphas.mul(radii).mul(-radii.sub(1))
    const shell = _alphas.div(dshell.add(eps))
    const kernel = tf.exp(_alphas.sub(shell)).mul(_weights).sum([-2])
    const dkernel = kernel.mean([0,1], true)

    return kernel.div(dkernel)
}

const normalise = (x:tf.Tensor, axis?:number[]) => {
    const mean = x.mean(axis, true)
    const std = x.mean(axis, true).squaredDifference(x).div(height * width).sqrt()

    return x.sub(mean).div(std)
}

const lenia = (world:tf.Tensor4D) => {

    const kernel:tf.Tensor4D = constructKernel(alphas, betas, weights).reshape([height, width, channels, 1])
    const acts = tf.depthwiseConv2d(world, kernel, [1,1], "same")
    const update = growth(acts.matMul(linear)).mul(dt)

    const noise = uniform.apply(world.shape)
    const nextWorld = world.add(update).add(noise)
    return normalise(nextWorld, [1,2])
}

// const world = uniform.apply([1,256,256,4])
// const nextWorld = lenia(world)
// nextWorld