<script>
    import * as tf from '@tensorflow/tfjs';
	import { model, StringToHashBucketFast } from '@tensorflow/tfjs';
    
    let height = 32;
    let width = 32;
    let mu = tf.scalar(.1)
    let sigma = tf.scalar(.015)

    const eps = tf.scalar(1e-12)


    const growth = (x:tf.Tensor) => {
        const [mean, std] = [mu.reshape([-1,1,1]), sigma.reshape([-1,1,1])]
        const top = x.sub(mean).square()
        const bottom = std.square().mul(2)
        
        const g = tf.exp(-top.div(bottom))

        return g.mul(2).sub(1)

    }

    const radius = (beta:tf.Tensor) => {
        const yi = tf.linspace(-height/2, height/2, height)
        const xi = tf.linspace(-width/2, width/2, width)
        const [yy, xx] = tf.meshgrid(yi, xi)
        
        const r = tf.sqrt(yy.square().add(xx.square()))
        const rd = beta.mul(r.max())

        const radius = r.div(rd).clipByValue(0,1)
        return radius
    }


    const constructShell = (alpha:tf.Tensor, beta:tf.Tensor, weight:tf.Tensor) => {
        let r = radius(beta)
        const rd = alpha.mul(r).mul(-r.sub(1))
        r = alpha.div(rd.add(eps))
        r = tf.exp(alpha.sub(r)).mul(weight)
        
        return r
    }
    
    const K = (alphas:tf.Tensor, betas:tf.Tensor, weights:tf.Tensor) => {
        let kernel = tf.zeros([height, width])
        const [n, ...rest] = alphas.shape

        for (let idx = 0; idx < n; idx++) {
            const shell = constructShell(alphas.slice([idx]), betas.slice([idx]), weights.slice([idx]))
            kernel.add(shell)
        }

        kernel = kernel.div(kernel.sum())

        return kernel
    }

    const input = tf.input({shape:[4,256,256]})

    const normalise = (x:tf.Tensor, axis?:number[]) => {
        const mean = x.mean(axis, true)
        const std = x.mean(axis, true).squaredDifference(x).div(height * width).sqrt()

        return x.sub(mean).div(std)
    }

    const lenia = (world:Tensor) => {
        const kernel = K(alphas, betas, weights).reshape([height, width, 1, -1])
        let update = tf.depthwiseConv2d(world, kernel, [1,1], "same")
        update = tf.einsum('io,ihw->ohw', linear, update)
        update = growth(update).mul(dt)

        const noise = tf.initializers.randomUniform(world.shape)
        const nextWorld = world.add(update).add(noise)

        return normalise(nextWorld)

    }
    



</script>