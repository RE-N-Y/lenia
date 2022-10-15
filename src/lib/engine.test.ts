import * as tf from "@tensorflow/tfjs";
import { lenia } from "./engine"
import { describe, expect, it } from 'vitest';

describe("lenia", () => {
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
        const image = engine.convert(world)

        expect(image.shape).toEqual([height, width])
    })
    
})