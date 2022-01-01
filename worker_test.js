importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js");
importScripts("my_gru_layer.js")

const batchSize = 1
const timesteps = 1
const size = 1
const shape = [timesteps, size]
const batchShape = [batchSize, ...shape]
const inputs = tf.input({batchShape})
const units = 16

let outputs = inputs
let state

;[outputs, state] = tf.layers.rnn({
    cell:[
        tf.layers.gruCell({units})
    ],
    stateful: true,
    returnSequences: true,
    returnState: true
}).apply(outputs)

const model = tf.model({inputs, outputs})
model.compile({
    optimizer: tf.train.adam(), 
    loss: 'meanSquaredError'
})
console.log(model.summary())

function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

const DEBUG = false
const SAME = false

let busy = false
let i = 0
let prevIsBlack = false

let tsInput = []
let tsLabel = []

let batchInput = []
let batchLabel = []

self.addEventListener('message', async e => {
    if (busy) return
    busy = true
    i++

    // const isBlack = i%3===0
    const isBlack = getRandomInt(0, 3) === 2
    if (SAME) prevIsBlack = isBlack 

    const x = isBlack ? tf.zeros([size]) : tf.ones([size])
    const y = prevIsBlack ? tf.zeros([units]) : tf.ones([units])

    if (!SAME) prevIsBlack = isBlack 

    tsInput.push(x)
    tsLabel.push(y)

    if (tsInput.length < timesteps) {
        busy = false
        return
    }

    batchInput.push(tf.stack(tsInput))
    batchLabel.push(tf.stack(tsLabel))

    tsInput = []
    tsLabel = []

    if (batchInput.length < batchSize) {
        busy = false
        return
    }

    if (DEBUG) console.log("batchInput = ", await tf.stack(batchInput).data())
    if (DEBUG) console.log("batchLabel = ", await tf.stack(batchLabel).data())

    const h = await model.fit(tf.stack(batchInput), tf.stack(batchLabel), {
        batchSize,
        epochs: 5,
        shuffle: false
    })

    batchInput = []
    batchLabel = []
    
    console.log("Loss: " + h.history.loss[0].toFixed(4))

    busy = false
})

