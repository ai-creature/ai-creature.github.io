importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js')
importScripts('agent_sac.js')
importScripts('reply_buffer.js')

;(async () => {
    const DISABLED = false

    const print = (...args) => console.log(...args)

    const agent = new AgentSac({batchSize: 10, verbose: true})
    await agent.init()
    await agent.checkpoint() // overwrite
    self.postMessage({weights: await Promise.all(agent.actor.getWeights().map(w => w.array()))}) // syncronize

    const rb = new ReplyBuffer(1000, ({ state: [telemetry, frame], action, reward }) => {
        frame.dispose()
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
            state: [telemetry, frame], 
            action, 
            reward, 
            nextState: [nextTelemetry, nextFrame] 
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
                state: [tf.stack(telemetries), tf.stack(frames)], 
                action: tf.stack(actions), 
                reward: tf.stack(rewards), 
                nextState: [tf.stack(nextTelemetries), tf.stack(nextFrames)]
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
        let { id, state: [telemetry, frames], action, reward, priority } = transition
    
        return tf.tidy(() => {
            state = [
                tf.tensor1d(telemetry),
                tf.tensor3d(frames, agent._frameStackShape)
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
