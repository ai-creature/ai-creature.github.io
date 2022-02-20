importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js')
importScripts('agent_sac.js')
importScripts('reply_buffer.js')

;(async () => {
    const DISABLED = false

    const agent = new AgentSac({batchSize: 100, verbose: true})
    await agent.init()
    await agent.checkpoint() // overwrite
    agent.actor.summary()
    self.postMessage({weights: await Promise.all(agent.actor.getWeights().map(w => w.array()))}) // syncronize

    const rb = new ReplyBuffer(50000, ({ state: [telemetry, frameL, frameR], action, reward }) => {
        frameL.dispose()
        frameR.dispose()
        telemetry.dispose()
        action.dispose()
        reward.dispose()
    })

    /**
     * Worker.
     * 
     * @returns delay in ms to get ready for the next job
     */
    const job = async () => {
// throw 'disabled'
        if (DISABLED) return 99999
        if (rb.size < agent._batchSize*10) return 1000
        
        const samples = rb.sample(agent._batchSize) // time fast
        if (!samples.length) return 1000
    
        const 
            framesL = [],
            framesR = [],
            telemetries = [],
            actions = [],
            rewards = [],
            nextFramesL = [],
            nextFramesR = [],
            nextTelemetries = []
    
        for (const {
            state: [telemetry, frameL, frameR], 
            action, 
            reward, 
            nextState: [nextTelemetry, nextFrameL, nextFrameR] 
        } of samples) {
            framesL.push(frameL)
            framesR.push(frameR)
            telemetries.push(telemetry)
            actions.push(action)
            rewards.push(reward)
            nextFramesL.push(nextFrameL)
            nextFramesR.push(nextFrameR)
            nextTelemetries.push(nextTelemetry)
        }
    
       tf.tidy(() => {
            console.time('train')
            agent.train({
                state:     [tf.stack(telemetries), tf.stack(framesL), tf.stack(framesR)],
                action:     tf.stack(actions), 
                reward:     tf.stack(rewards), 
                nextState: [tf.stack(nextTelemetries), tf.stack(nextFramesL), tf.stack(nextFramesR)]
            })
            console.timeEnd('train')
        })

        console.time('train postMessage')
        self.postMessage({
            weights: await Promise.all(agent.actor.getWeights().map(w => w.array()))
        })
        console.timeEnd('train postMessage')
    
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
        let { id, state: [telemetry, frameL, frameR], action, reward, priority } = transition
    
        return tf.tidy(() => {
            state = [
                tf.tensor1d(telemetry),
                tf.tensor3d(frameL, agent._frameStackShape),
                tf.tensor3d(frameR, agent._frameStackShape)
            ]
            action = tf.tensor1d(action)
            reward = tf.tensor1d([reward])
    
            return { id, state, action, reward, priority }
        })
    }
    
    let i = 0
    self.addEventListener('message', async e => {
        i++

        if (DISABLED) return
        if (i%50 === 0) console.log('RBSIZE: ', rb.size)
    
        switch (e.data.action) {
            case 'newTransition':
                const transition = decodeTransition(e.data.transition)
                rb.add(transition)

                tf.tidy(()=> {
                    return
                    const {
                        state: [telemetry, frameL, frameR], 
                        action,
                    } = transition;
                    const state = [tf.stack([telemetry]), tf.stack([frameL]), tf.stack([frameR])]
                    const q1TargValue = agent.q1Targ.predict([...state, tf.stack([action])], {batchSize: 1})
                    const q2TargValue = agent.q2Targ.predict([...state, tf.stack([action])], {batchSize: 1})                    
                    console.log('value', Math.min(q1TargValue.arraySync()[0][0], q2TargValue.arraySync()[0][0]).toFixed(5))
                })


                break
            default:
                console.warn('Unknown action')
                break
        }
    
        if (i % rb._limit === 0)
            agent.checkpoint() // timer ~ 500ms, don't await intentionally
    })
})()
