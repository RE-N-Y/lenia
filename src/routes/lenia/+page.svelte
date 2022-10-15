<script lang="ts">
    import { onMount } from 'svelte';
    import { browser } from '$app/environment';
    import * as tf from "@tensorflow/tfjs";
    import { lenia } from "$lib/engine";
	

    let [height, width] = [384, 384]
    let canvas:HTMLCanvasElement;
    let last = { time:0, frame:0 }
    let current = { time:0, frame:0 }
    let fps = 0;

    onMount(() => {
        let ctx = canvas.getContext("2d")
        ctx?.fillRect(0, 0, 256, 256)
    })

    if (browser) { 
        let tick = 0;
        const duration = 1000;
        const engine = lenia(16, 16, 3, 4)
        
        let nextWorld;
        let world = tf.randomUniform([1,height,width,4])
        world = engine.smooth(engine.smooth(world))

        const simulate = async () => {
            if (tick < duration) {
                requestAnimationFrame(simulate)
                
                current = { time:performance.now(), frame:tick } 
                fps = 1000 * (current.frame - last.frame) / (current.time - last.time)
                last = { ...current }
            }

            nextWorld = engine.run(world)
            world.dispose()
            world = nextWorld
            
            const pixels = engine.convert(world)
            await tf.browser.toPixels(pixels, canvas)
            pixels.dispose()
            tick++
        }

        requestAnimationFrame(simulate)
        
    }
</script>


<div>
    <canvas bind:this={canvas} height={height} width={width}/>
    <p>fps:{Math.round(fps)}</p>
</div>