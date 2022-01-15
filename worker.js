importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js")
importScripts("agent_sac.js")
importScripts("reply_buffer.js")

const [TRUE, FALSE] = [true, false]

const agent = new AgentSac({batchSize: 7})

const rb = new ReplyBuffer(100)

const SAME = FALSE
let busy = TRUE
let i = 0
let prevIsBlack = FALSE
let stack = []
let stack2 = []

const job = async () => {
    const samples = rb.sample(agent._batchSize)
    if (!samples.length) return

    const 
        states = [],
        actions = [],
        rewards = [],
        nextStates = []

    for (const { state, action, reward, nextState } of samples) {
        console.log(reward.shape)
        states.push(state)
        actions.push(action)
        rewards.push(reward)
        nextStates.push(nextState)
    }

    tf.tidy(() => {
        
        agent.learn({
            state: tf.stack(states), 
            action: tf.stack(actions), 
            reward: tf.stack(rewards), 
            nextState: tf.stack(nextStates)
        })
    })
}

const tick = async () => {
    try {
        await job()
        setTimeout(tick, 0)
    } catch (e) {
        console.error(e)
        setTimeout(tick, 5000) // show must go on (҂◡_◡) ᕤ
    }
}
setTimeout(tick, 0)

const decodeTransition = transition => {
    let { id, state, action, reward } = transition

    return tf.tidy(() => {
        // TODO: figure out which type is better 'float32' for tf.tensor
        // TODO: make sexy getters for shapes
        state = tf.tensor3d(state, [...agent._frameShape.slice(0, 2), agent._frameShape[2] * agent._nFrames])
        action = tf.tensor1d(action)
        reward = tf.scalar(reward)

        return { id, state, action, reward }
    })
}

self.addEventListener('message', async e => {
    switch (e.data.action) {
        case 'newTransition':
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



