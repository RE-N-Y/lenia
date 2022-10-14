import * as tf from "@tensorflow/tfjs";
import { lenia } from "./engine"
import { describe, expect, it } from 'vitest';

describe("lenia", () => {
    const shape = [1,256,256,4]

    it("should not leak memory", () => {
        
        let nextWorld;
        let world = tf.randomUniform(shape)
        
        const last = tf.memory()
        
        nextWorld = lenia(world)
        world.dispose()
        world = nextWorld
        
        const current = tf.memory()
        
        expect(last.numTensors).toEqual(current.numTensors)
    })

    it("should maintain its shape", () => {    
        let world = tf.randomUniform(shape)
        world = lenia(world)
        expect(world.shape).toEqual(shape)
    })
    
})