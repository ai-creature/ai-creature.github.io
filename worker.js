importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js")
importScripts("agent_sac.js")
importScripts("reply_buffer.js")

const [TRUE, FALSE] = [true, false]

const agent = new AgentSac({batchSize: 3})

const rb = new ReplyBuffer()

const SAME = FALSE
let busy = TRUE
let i = 0
let prevIsBlack = FALSE
let stack = []
let stack2 = []

/**
 * Worker.
 * 
 * @returns delay in ms to get ready for the next job
 */
const DISABLED = FALSE

const job = async () => {
    if (DISABLED) return
    const samples = rb.sample(agent._batchSize)
    if (!samples.length) return 1000

    const 
        frames = [],
        telemetries = [],
        actions = [],
        rewards = [],
        nextFrames = [],
        nextTelemetries = []

    for (const { 
        state: [frame, telemetry], 
        action, 
        reward, 
        nextState: [nextFrame, nextTelemetry] 
    } of samples) {
        frames.push(frame)
        telemetries.push(telemetry)
        actions.push(action)
        rewards.push(reward)
        nextFrames.push(nextFrame)
        nextTelemetries.push(nextTelemetry)
    }

    tf.tidy(() => {
        agent.learn({
            state: [tf.stack(frames), tf.stack(telemetries)], 
            action: tf.stack(actions), 
            reward: tf.stack(rewards), 
            nextState: [tf.stack(nextFrames), tf.stack(nextTelemetries)]
        })
    })
    
    self.postMessage({weights: await Promise.all(agent.actor.getWeights().map(w => w.array()))})

    return 1
}

/**
 * Executes job.
 */
const tick = async () => {
    try {
        setTimeout(tick, await job())
    } catch (e) {
        console.error(e)
        setTimeout(tick, 5000) // show must go on (҂◡_◡) ᕤ
    }
}

setTimeout(tick, 1000)

/**
 * Decode transition from the main thread.
 * 
 * @param {{ id, state, action, reward }} transition 
 * @returns 
 */
const decodeTransition = transition => {
    let { id, state: [frames, telemetry], action, reward } = transition

    return tf.tidy(() => {
        state = [
            tf.tensor3d(frames, agent._frameStackShape),
            tf.tensor1d(telemetry)
        ]
        action = tf.tensor1d(action)
        reward = tf.tensor1d([reward])

        return { id, state, action, reward }
    })
}

self.addEventListener('message', async e => {
    switch (e.data.action) {
        case 'newTransition':
            if (DISABLED) return
            // if (rb.size == 0) console.time('RB FULL')
            // if (rb.size == rb._limit-1) {console.timeEnd('RB FULL'); console.log(cnt)}
            rb.add(decodeTransition(e.data.transition))
            break
        default:
            console.warn('Unknown action')
            break
    }

    return

    if (busy || i > 50) return
    busy = TRUE
    i++

    const frame = tf.tensor3d(e.data, shape, 'float32')
    const frameNorm = frame.div(tf.scalar(255))

    stack.push(frameNorm)
    stack2.push(frame)

    if (stack.length < stackFrames) {
        busy = FALSE
        return
    }

    const input = tf.stack([tf.concat(stack, 2)])

    // console.time("learn timer")
    // tf.tidy(()=>{
    //     agent.learn(input)
    // })
    // console.timeEnd("learn timer")


    // const [action, logProb] = agent.sampleAction(input)
    // console.log("ACTION: ", logProb)
    // console.log("ARRAY: ", await logProb.array())



    //learn(st)
    
    // const lossFunction = () => tf.tidy(() => {
    //     const preds = model.predict(input)

    //     return tf.losses.meanSquaredError(labels, preds).asScalar()
    // })

    // const {value, grads} = tf.variableGrads(lossFunction)

    // optimizer.applyGradients(grads)
     
    // console.log("Loss: " + value)
    
    // tf.dispose(action)
    // tf.dispose(logProb)
    // tf.dispose(value)
    // tf.dispose(grads)
    input.dispose()
    // labels.dispose()

    const data = await frame.array()

    stack.forEach(frameNorm => frameNorm.dispose())
    stack2.forEach(frame => frame.dispose())
    stack = []
    stack2 = []

    // if (i%24 < 5) {
        console.time("send weights")
        self.postMessage({frame: data, weights: agent.actor.getWeights().map(w => w.arraySync())}) // timer ~10ms for send Weights

        console.timeEnd("send weights")
    // }

    busy = FALSE
})



