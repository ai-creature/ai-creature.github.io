importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js")
importScripts("agent_sac.js")

const [TRUE, FALSE] = [true, false]

const shape = [128, 256, 3]
const stackFrames = 1
const batchSize = 1
const outputUnits = 1

const inputs = tf.input({batchShape : [batchSize, ...shape.slice(0, 2), shape[2]*stackFrames]})
let outputs = inputs

const agent = new AgentSac({})

const SAME = FALSE
let busy = TRUE
let i = 0
let prevIsBlack = FALSE
let stack = []
let stack2 = []

self.addEventListener('message', async e => {
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

