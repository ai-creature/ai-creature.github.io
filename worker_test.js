importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js");
importScripts("my_gru_layer.js")

const batchSize = 1
const timesteps = 1
const shape = [timesteps, 1]
const batchShape = [batchSize, ...shape]
const inputs = tf.input({batchShape})
const units = 16

let outputs = inputs

outputs = tf.layers.rnn({
    cell:[
        tf.layers.gruCell({units})
    ],
    stateful: true,
    returnSequences: true,
    returnState: false
}).apply(outputs)

// outputs = tf.layers.dense({units}).apply(outputs)

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

let busy = false
let i = 0
let prevIsBlack = false
let batchInput = []
let batchLabel = []
self.addEventListener('message', async e => {
    if (busy) return
    busy = true
    // i++

    // if (Math.random() < 0.05) {
    // if (i%(batchSize*10) === 0) {
    //     console.log("***")
    //     model.resetStates()
    // }

    // const randIndex = getRandomInt(0, timesteps - 1)
    // const blackIndex = (randIndex + 1) % timesteps
    // console.log("randIndex = ", randIndex, (randIndex + 1) % timesteps)

    let blackIndex
    const inputArray = new Array(timesteps).fill().map((el, ind) => {
        i++
        if (i%(timesteps + 2) === 0) {
            blackIndex = ind
            return 1
        } else
            return 0
    })

    let blackIndexLabel
    if (prevIsBlack) {
        blackIndexLabel = 0
        prevIsBlack = false
    } else if (blackIndex !== undefined && (blackIndex + 1) % timesteps === 0) {
        prevIsBlack = true
    } else if (blackIndex !== undefined) {
        blackIndexLabel = blackIndex + 1
    }

    // if (blackIndex !== undefined) {
    //     blackIndexLabel = blackIndex
    // }

    // inputArray[randIndex] = 1
    const input = tf.tensor(inputArray, shape)
    
    // const res = model.predict(input, {batchSize})
    
    const outputArray = new Array(timesteps).fill().map(el => new Array(units).fill(1))
    if (blackIndexLabel !== undefined) 
        outputArray[blackIndexLabel] = new Array(units).fill(0)
    const label = tf.tensor(outputArray, [shape[0], units])

        
    if (DEBUG) console.log("input = ", inputArray, input.shape)
    if (DEBUG) console.log("label = ", outputArray, label.shape)

    batchInput.push(input)
    batchLabel.push(label)

    if (batchInput.length < batchSize) {
        busy = false
        return
    }

    if (DEBUG) console.log("batchInput = ", tf.stack(batchInput).shape)
    if (DEBUG) console.log("batchLabel = ", tf.stack(batchLabel).shape)

    const h = await model.fit(tf.stack(batchInput), tf.stack(batchLabel), {
        batchSize,
        epochs: 5,
        shuffle: false
    })

    batchInput = []
    batchLabel = []
    
    console.log("Loss: " + h.history.loss[0].toFixed(2))

    busy = false
})

