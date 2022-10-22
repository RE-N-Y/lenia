<script lang="ts">
    import * as tf from "@tensorflow/tfjs";
    import { lenia, smooth, convert, pdf } from "$lib/engine"
    import lock from "@material-design-icons/svg/filled/lock.svg"

    let channels = 4;
    let bumps = 3;
    let [kh,kw] = [32,32]
    let speed = .1
    let engine = lenia(kh, kw, bumps, channels);

    let play = false;

    let [sigmaX, sigmaY] = [64, 64]
    let [height, width] = [256, 256]
    let canvas:HTMLCanvasElement;
    let last = { time:0, frame:0 }
    let current = { time:0, frame:0 }
    let fps = 0;
    let world = tf.zeros([1,height,width,channels]);
    
    $: [_, wh, ww, wc] = world.shape
    $: sync = (wh === height) && (ww === width) && (wc === channels)

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
        
        const nextWorld = tf.tidy(() => {
            const yi = tf.linspace(0, height, height)
            const xi = tf.linspace(0, width, width)
            const [yy, xx] = tf.meshgrid(yi, xi)
            const brush = pdf(yy, xx, mu, [sigmaX, sigmaY]).reshape([1,height,width,1]).tile([1,1,1,channels])
            const result = world.add(brush)

            return result
        })

        updateWorld(nextWorld)
        redraw()
    }

    let counter = 0;
    const simulate = async () => {
        if (play) {
            requestAnimationFrame(simulate)
            current = { time:performance.now(), frame:counter } 
            fps = 1000 * (current.frame - last.frame) / (current.time - last.time)
            last = { ...current }
        }
    
        updateWorld(engine.run(world, speed))
        redraw()
        counter++
    }

    const reset = () => {
        updateWorld(tf.randomUniform([1,height,width,channels]))
        resetEngine()
        updateWorld(engine.run(world, speed))
        redraw()
    }

    const handleClear = async () => {
        updateWorld(tf.zeros(world.shape))
        redraw()
    }

    const resetEngine = () => {
        engine.dispose()
        engine = lenia(kh, kw, bumps, channels)
    }
    
    const handleSmooth = () => {
        updateWorld(smooth(world))
        redraw()
    }

    const handlePlay = async () => {
        play = !play;
        if (!sync) reset()
        requestAnimationFrame(simulate)
    }
        
</script>

<div class="container mx-auto p-4">
    <div class="text-center">
        <h1>Neural Celluar Automata</h1>
        <p>Generating artificial life with Lenia</p>
    </div>
    <div class="flex flex-col items-center">
        <div class={`h-[${height}px] w-[${width}px shadow-md]`} on:mousedown={handleMouseDown}>
            <span class="text-xs p-1">fps {Math.round(fps)}</span>
            <canvas class="rounded bg-clip-border bg-black" bind:this={canvas} height={`${height}`} width={`${width}`} />
        </div>
        <button class="m-2" on:click={handlePlay}>{play ? "Stop" : "Play"}</button>
    </div>
    <div class="flex justify-center">
        <div class="space-y-6">
            <div class="space-y-2">
                <h2 class="mb-1">Engine</h2>
                <div class="flex space-x-4">
                    <div>
                        <label class="block text-sm text-black/70" for="kh">Kernel Height</label>
                        <input class="shadow appearance-none rounded-md w-32 px-2" id="kh" type="number" bind:value={kh}/>
                    </div>
                    <div>
                        <label class="block text-sm text-black/70" for="kw">Kernel Width</label>
                        <input class="shadow appearance-none rounded-md w-32 px-2" id="kw" type="number" bind:value={kw}/>
                    </div>
                    <div>
                        <label class="block text-sm text-black/70" for="bumps">Bumps</label>
                        <input class="shadow appearance-none rounded-md w-32 px-2" id="bumps" type="range" bind:value={bumps} min="1" max="5"/>
                    </div>
                    <div>
                        <label class="block text-sm text-black/70" for="speed">Speed</label>
                        <input class="shadow appearance-none rounded-md w-32 px-2" id="speed" type="range" min="0.1" max="1" step="0.1" bind:value={speed}/>
                    </div>
                </div>    
                <button on:click={resetEngine}>Reset engine</button>
            </div>

            <div class="flex space-x-8">
                <div>
                    <h2 class="mb-1">World</h2>
                    <div class="relative">
                        <div class={`${!play && "hidden"} flex flex-col justify-center items-center absolute w-full h-full`}>
                            <img src={lock} alt="lock"/>
                            <p>Locked</p>
                        </div>
                        <div class="flex space-x-4 mb-2">
                            <div class={`${play && "opacity-25"}`}>
                                <label class="block text-sm text-black/70" for="height">Height</label>
                                <input class="shadow appearance-none rounded-md w-32 px-2" id="height" type="number" bind:value={height} disabled={play} />
                            </div>
                            <div class={`${play && "opacity-25"}`}>
                                <label class="block text-sm text-black/70" for="width">Width</label>
                                <input class="shadow appearance-none rounded-md w-32 px-2" id="width" type="number" bind:value={width} disabled={play} />
                            </div>
                        </div>
                        <div class="flex space-x-4 mb-2">
                            <div class={`${play && "opacity-25"}`}>
                                <label class="block text-sm text-black/70" for="channels">Channels</label>
                                <input class="shadow appearance-none rounded-md w-32 px-2" id="channels" type="range" bind:value={channels} min="1" max="8" disabled={play}/>
                            </div>
                        </div>
                    </div>
                    <button on:click={reset}>New world</button>
                    <button on:click={handleClear}>Clear world</button>
                    <button on:click={handleSmooth}>Smooth world</button>
                </div>
                
                <div>
                    <h2 class="mb-1">Editing</h2>
                    <div class="flex space-x-4 mb-2">
                        <div>
                            <label class="block text-sm text-black/70" for="sigmaX">x</label>
                            <input class="shadow appearance-none rounded-md w-32 px-2" id="sigmaX" type="number" bind:value={sigmaX}/>
                        </div>
                        <div>
                            <label class="block text-sm text-black/70" for="sigmaY">y</label>
                            <input class="shadow appearance-none rounded-md w-32 px-2" id="sigmaY" type="number" bind:value={sigmaY}/>
                        </div>
                    </div>
                </div>
            </div>
            
        </div>
        
    </div>
</div>