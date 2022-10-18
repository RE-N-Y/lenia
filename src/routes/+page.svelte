<script lang="ts">
    import * as tf from "@tensorflow/tfjs";
    import { lenia, smooth, convert, pdf } from "$lib/engine"

    const engine = lenia(32, 32, 3, 4);

    let play = false;

    let [height, width] = [256, 256]
    let canvas:HTMLCanvasElement;
    let last = { time:0, frame:0 }
    let current = { time:0, frame:0 }
    let fps = 0;

    let world = tf.zeros([1,height,width,4])

    const redraw = async () => {
        const pixels = convert(world)
        await tf.browser.toPixels(pixels, canvas)
        pixels.dispose()
    }

    const updateWorld = (nextWorld:tf.Tensor) => {
        let tmp = nextWorld
        world.dispose()
        world = tmp
    }

    const handleMouseDown = async (event:MouseEvent) => {
        const rectangle = canvas.getBoundingClientRect()
        const mu = [event.clientX - rectangle.x, event.clientY - rectangle.y]
        const sigma = [50, 50]
        
        const nextWorld = tf.tidy(() => {
            const yi = tf.linspace(0, height, height)
            const xi = tf.linspace(0, width, width)
            const [yy, xx] = tf.meshgrid(yi, xi)
            const brush = pdf(yy, xx, mu, sigma).reshape([1,height,width,1]).tile([1,1,1,4])
            const result = world.add(brush)

            return result
        })

        updateWorld(nextWorld)
        redraw()
    }

    let tick = 0;
    const simulate = async () => {
        if (play) {
            requestAnimationFrame(simulate)
            current = { time:performance.now(), frame:tick } 
            fps = 1000 * (current.frame - last.frame) / (current.time - last.time)
            last = { ...current }
        }
    
        updateWorld(engine.run(world))
        redraw()
        tick++
    }

    const handlePlay = () => { 
        requestAnimationFrame(simulate)
        play = !play 
    }

    const handleReset = () => {
        updateWorld(tf.randomUniform(world.shape))
        updateWorld(engine.run(smooth(smooth(world))))
        redraw()
    }

    const handleClear = () => {
        updateWorld(tf.zeros(world.shape))
        redraw()
    }
        
        
</script>


<div class="container mx-auto">
    <div class="p-2">
        <h1>Neural Celluar Automata</h1>
        <p>Generating artificial life with Lenia</p>
    </div>
    
    <div class="flex p-2">
        <div class="flex-auto">
            <div class={`h-[${height}px] w-[${width}px shadow-md]`} on:mousedown={handleMouseDown}>
                <canvas class="rounded bg-clip-border bg-black" bind:this={canvas} height={`${height}`} width={`${width}`} />
            </div>
            <div class="space-x-1">
                <button on:click={handlePlay}>{play ? "stop" : "play"}</button>
                <button on:click={handleReset}>reset</button>
                <button on:click={handleClear}>clear</button>
            </div>
            
            <p>fps:{Math.round(fps)}</p>
        </div>
        <div class="flex-auto">

        </div>
        
    </div>
</div>