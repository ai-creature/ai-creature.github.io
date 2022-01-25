importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js')
importScripts('agent_sac.js')
importScripts('reply_buffer.js')

;(async () => {
    const print = (...args) => console.log(...args)

    const agent = new AgentSac({batchSize: 100, verbose: true})
    await agent.init()
    await agent.checkpoint() // overwrite

    const rb = new ReplyBuffer(1000, ({ state: [frame, telemetry], action, reward }) => {
        frame.dispose()
        telemetry.dispose()
        action.dispose()
        reward.dispose()
    })
    
    const DISABLED = false
    
    /**
     * Worker.
     * 
     * @returns delay in ms to get ready for the next job
     */
    const job = async () => {
        if (DISABLED) return 99999
        // if (rb.size < 500) return 1000
    
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
            agent.train({
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
            // setTimeout(tick, 5000) // show must go on (҂◡_◡) ᕤ
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
    
    let i = 0
    self.addEventListener('message', async e => {
        i++

        if (DISABLED) return
        if (i%20 === 0) console.log('RBSIZE: ', rb.size)
    
        switch (e.data.action) {
            case 'newTransition':
                rb.add(decodeTransition(e.data.transition))
                break
            default:
                console.warn('Unknown action')
                break
        }
    
        if (i % rb._limit === 0)
            agent.checkpoint() // timer ~ 500ms, don't await intentionally
    })
})()
