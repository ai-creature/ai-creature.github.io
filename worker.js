importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js")
// import * as tfvis from '@tensorflow/tfjs-vis';

const [TRUE, FALSE] = [true, false]

const shape = [128, 256, 3]
const stackFrames = 1
const batchSize = 1
const outputUnits = 1

const inputs = tf.input({batchShape : [batchSize, ...shape.slice(0, 2), shape[2]*stackFrames]})
let outputs = inputs

function getConvEncoder(outputs) {
    const kernelSize = [3, 3]
    const poolSize = 2
    const strides = 1
    const padding = 'same'
    const layers = 14
    
    let filterPow = 2
    
    for (let i = 0; i < layers; i++) {
        if (i%3 == 1) 
            filterPow++

        outputs = tf.layers.conv2d({
            filters: 2**filterPow,
            kernelSize,
            strides,
            padding,
            activation: 'relu',
            kernelInitializer: 'heNormal',
            biasInitializer: 'heNormal',
            trainable: true
        }).apply(outputs)
    
        if (i%2 == 1) 
            outputs = tf.layers.maxPooling2d({poolSize}).apply(outputs)
    }

    return outputs
}

outputs = getConvEncoder(outputs)
outputs = tf.layers.flatten().apply(outputs)
outputs = tf.layers.dense({units: outputUnits, activation: 'linear'}).apply(outputs)

const optimizer = tf.train.adam()
const model = tf.model({inputs, outputs})
console.log(model.summary())

let busy = TRUE
let i = 0
let prevIsBlack = FALSE

const SAME = FALSE

let stack = []
let stack2 = []

self.addEventListener('message', async e => {
    if (busy) return
    busy = TRUE
    i++

    // const isBlack = Math.random() <= 0.5
    const isBlack = i%3 === 0
    const val = isBlack ? 0 : 255

    const shift = 50
    const side = 12
    for (let i = shift; i < shift + side; i++) {
        for (let j = shift; j < shift + side; j++) {
            e.data[i][j] = [val, val, val]
        }
    }

    const frame = tf.tensor3d(e.data, shape, 'float32')
    const frameNorm = frame.div(tf.scalar(255))

    stack.push(frameNorm)
    stack2.push(frame)

    if (stack.length < stackFrames) {
        busy = FALSE
        return
    }

    if (i%64 === 0) {
        console.log("***")
        model.resetStates()
    }

    const input = tf.stack([tf.concat(stack, 2)])

    if (SAME) prevIsBlack = isBlack
    const labels = prevIsBlack ? tf.ones([1, outputUnits]) : tf.zeros([1, outputUnits])
    if (!SAME) prevIsBlack = isBlack
    
    const lossFunction = () => tf.tidy(() => {
        const preds = model.predict(input)

        return tf.losses.meanSquaredError(labels, preds).asScalar()
    })

    const {value, grads} = tf.variableGrads(lossFunction)

    optimizer.applyGradients(grads)
     
    console.log("Loss: " + value)
    
    tf.dispose(value)
    tf.dispose(grads)
    input.dispose()
    labels.dispose()

    const data = await frame.array()

    stack.forEach(frameNorm => frameNorm.dispose())
    stack2.forEach(frame => frame.dispose())
    stack = []
    stack2 = []

    self.postMessage(data)

    busy = FALSE
})

