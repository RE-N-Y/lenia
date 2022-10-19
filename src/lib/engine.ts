import * as tf from '@tensorflow/tfjs';

const eps = tf.scalar(4e-8)

export const pdf = (xi:tf.Tensor, yi:tf.Tensor, mu:number[], sigma:number[]) => {
    const result = tf.tidy(() => {
        const [muX, muY] = mu
        const [sigmaX, sigmaY] = sigma

        const dx = xi.sub(muX).square().div(sigmaX)
        const dy = yi.sub(muY).square().div(sigmaY)
        const p = tf.exp(dx.add(dy).div(-2))
        return p
    })

    return result
} 

const radius = (kh:number, kw:number) => {
    const result = tf.tidy(() => {
        const yi = tf.linspace(-kh/2, kh/2, kh)
        const xi = tf.linspace(-kw/2, kw/2, kw)
        const [yy, xx] = tf.meshgrid(yi, xi)
        const r = tf.sqrt(yy.square().add(xx.square()))

        return r
    })

    return result
}

// tensor to image
export const convert = (x:tf.Tensor4D) => tf.tidy(() => x.mean([-1,0]).clipByValue(0,1))

// smooth channels
const filter = tf.tensor([1,3,3,1])
export const smooth = (x:tf.Tensor4D) => {
    const result = tf.tidy(() => {
        const channels = x.shape[3]
        const kernel:tf.Tensor2D = filter.reshape([1,4]).mul(filter.reshape([4,1]))
        const nkernel:tf.Tensor4D = kernel.div(kernel.sum()).reshape([4,4,1,1]).tile([1,1,channels,1])
        const smooooth = tf.depthwiseConv2d(x, nkernel, [1,1], "same")
        return smooooth
    })

    return result
}

const normalise = (x:tf.Tensor, axis?:number[]) => {
    const result = tf.tidy(() => {
        const mean = x.mean(axis, true)
        const variance = mean.squaredDifference(x).mean(axis, true)
        const normal = x.sub(mean).div(variance.sqrt())

        return normal
    })

    return result
}

const growth = (x:tf.Tensor, mu:tf.Tensor, sigma:tf.Tensor) => {
    const result = tf.tidy(() => {
        const top = x.sub(mu).square()
        const bottom = sigma.square().mul(2)
        const g = tf.exp(top.mul(-1).div(bottom))
        const gg = g.mul(2).sub(1)

        return gg
    })

    return result

}

export const lenia = (kh:number, kw:number, bumps:number, channels:number) => {
    
    // growth parameters
    const mu = tf.variable(tf.randomUniform([channels]))
    const sigma = tf.variable(tf.randomUniform([channels]).mul(.1))
    
    // filter parameters
    const alphas = tf.variable(tf.randomUniform([channels, bumps]).mul(4.))
    const betas = tf.variable(tf.randomUniform([channels, bumps]))
    const weights = tf.variable(tf.randomUniform([channels, bumps]))
    const linear = tf.variable(tf.randomUniform([channels, channels], -1/Math.sqrt(channels), 1/Math.sqrt(channels)))

    const constructKernel = (alphas:tf.Tensor, betas:tf.Tensor, weights:tf.Tensor) => {

        const result = tf.tidy(() => {
            const [_alphas, _betas, _weights] = [alphas, betas, weights].map(t => t.reshape([1,1,bumps,channels]))
            const r = radius(kh, kw).reshape([kh, kw, 1, 1]).tile([1,1,bumps,channels])
            const rd = _betas.mul(r.max([0,1], true))
            const radii = r.div(rd).clipByValue(0,1)    
    
            const dshell = _alphas.mul(radii).mul(radii.sub(1)).mul(-1)
            const shell = _alphas.div(dshell.add(eps))
            const kernel = tf.exp(_alphas.sub(shell)).mul(_weights).sum([-2])
            const dkernel = kernel.sum([0,1], true)
            const K:tf.Tensor4D = kernel.div(dkernel).reshape([kh, kw, channels, 1])
    
            return K
        })
    
        return result
    }

    const run = (world:tf.Tensor4D, speed:number=.1) => {
        const result = tf.tidy(() => {
            const kernel = constructKernel(alphas, betas, weights, kh, kw, bumps, channels)
            const acts = tf.depthwiseConv2d(world, kernel, 1, "same").matMul(linear)
            const update = growth(acts, mu, sigma).mul(speed)
            const noise = tf.randomUniform(world.shape).mul(.001)
            const nextWorld = normalise(world.add(update).add(noise), [1,2])    

            return nextWorld
        })

        return result
    }

    return { run }
}

    

// const [height, width] = [256, 256]
// const world:tf.Tensor4D = tf.randomUniform([1, height, width, channels])
// const nextWorld = lenia(world)