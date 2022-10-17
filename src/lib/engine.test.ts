import * as tf from "@tensorflow/tfjs";
import { lenia, convert, pdf } from "./engine"
import { describe, expect, it } from 'vitest';

describe("lenia engine", () => {
    const [height, width] = [64,64]
    const [kh, kw] = [8,8]
    const [bumps, channels] = [3,4]
    const shape = [1, height, width, channels]
    const engine = lenia(kh, kw, bumps, channels)

    it("should not leak memory", () => {
        let nextWorld;
        let world = tf.randomUniform(shape)
        
        const last = tf.memory()
        
        nextWorld = engine.run(world)
        world.dispose()
        world = nextWorld
        
        const current = tf.memory()
        
        expect(last.numTensors).toEqual(current.numTensors)
    })

    it("should maintain its shape", () => {    
        let world = tf.randomUniform(shape)
        world = engine.run(world)

        expect(world.shape).toEqual(shape)
    })

    it("should convert tensors to images", () => {
        const world = tf.randomUniform(shape)
        const image = convert(world)

        expect(image.shape).toEqual([height, width])
    })
})

describe("pdf", () => {
    const [height,width] = [32,32]
    const [muX, muY] = [16,16]
    const [sigmaX, sigmaY] = [7,7]

    const yi = tf.linspace(0, height, height)
    const xi = tf.linspace(0, width, width)
    const [yy, xx] = tf.meshgrid(yi, xi)
    const eps = .1

    it("should sum upto normalisation constant", async () => {
        const p = pdf(yy, xx, [muX,muY], [sigmaX,sigmaY])
        const [sum] = await p.sum().data()
        const constant = 2 * Math.PI * Math.sqrt(sigmaX * sigmaY)
        expect(sum / constant).gt(1 - eps).lt(1 + eps)
    })

    it("should not leak memory", () => {
        const last = tf.memory()
        
        const p = pdf(yy, xx, [muX,muY], [sigmaX,sigmaY])
        p.dispose()
        
        const current = tf.memory()

        expect(last.numTensors).toEqual(current.numTensors)
    })
})