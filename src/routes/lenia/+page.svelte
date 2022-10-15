<script>
    import { browser } from '$app/environment';
    import * as tf from "@tensorflow/tfjs";
    import { convert, lenia, smooth } from "$lib/engine";

    let last = { time:0, frame:0 }
    let current = { time:0, frame:0 }
    let fps = 0

    if (browser) { 
        let tick = 0;
        const duration = 1000;
        const canvas = document.getElementById("world")
        
        let nextWorld;
        let world = tf.randomUniform([1,512,512,4])
        world = smooth(smooth(world))

        const simulate = async () => {
            if (tick < duration) {
                current = { time:requestAnimationFrame(simulate), frame:tick } 
                fps = (current.frame - last.frame) / (current.time - last.time)
                last = { ...current }
            }

            nextWorld = lenia(world)
            world.dispose()
            world = nextWorld
            
            const pixels = convert(world)
            await tf.browser.toPixels(pixels, canvas)
            pixels.dispose()
            tick++
        }

        requestAnimationFrame(simulate)
        
    }
    
    


</script>


<div>
    <canvas id="world"/>
    <p>fps:{fps}</p>
    <p>frame:{current.frame} time:{current.time}</p>
</div>