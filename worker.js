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
let busy = FALSE
let i = 0
let prevIsBlack = FALSE
let stack = []
let stack2 = []

self.addEventListener('message', async e => {
    if (busy) return
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

    const [action, logExpr] = agent.sampleAction(input)

    console.log("ACTION: ", logExpr)
    console.log("ARRAY: ", await logExpr.array())
    
    // const lossFunction = () => tf.tidy(() => {
    //     const preds = model.predict(input)

    //     return tf.losses.meanSquaredError(labels, preds).asScalar()
    // })

    // const {value, grads} = tf.variableGrads(lossFunction)

    // optimizer.applyGradients(grads)
     
    // console.log("Loss: " + value)
    
    tf.dispose(action)
    tf.dispose(logExpr)
    // tf.dispose(value)
    // tf.dispose(grads)
    input.dispose()
    // labels.dispose()

    const data = await frame.array()

    stack.forEach(frameNorm => frameNorm.dispose())
    stack2.forEach(frame => frame.dispose())
    stack = []
    stack2 = []

    self.postMessage(data)

    busy = FALSE
})

