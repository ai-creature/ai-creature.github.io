/*
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

const xs = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]);
const ys = tf.tensor2d([1, 3, 5, 7, 8], [5, 1]);

await model.fit(xs, ys, {epochs: 100});
model.predict(tf.tensor2d([6], [1, 1])).print(); 
*/

importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.12.0/dist/tf.min.js");

// const model = tf.sequential();
const shape = [128, 256, 3]
const batchSize = 1
const kernelSize = [3, 3]
const poolSize = 2
const strides = 1
const padding = 'same'
let filterPow = 3

const inputs = tf.input({
    // shape, 
    batchShape : [batchSize, ...shape]
})

layers = 14
let outputs = inputs
for (let i = 0; i < layers; i++) {
    if (i%3 == 1) {
        filterPow++
        
    }

    outputs = tf.layers.conv2d({
        filters: 2**filterPow,
        kernelSize,
        strides,
        padding,
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }).apply(outputs)

    if (i%2 == 1) {
        outputs = tf.layers.maxPooling2d({
            poolSize
        }).apply(outputs)
    }
}

console.log(outputs.shape)
outputs = tf.layers.reshape({targetShape: [1, 512]}).apply(outputs)
// outputs = tf.layers.flatten().apply(outputs)
console.log(outputs.shape)
// outputs = tf.layers.dense({units: 256}).apply(outputs)
// outputs = tf.layers.dense({units: 256}).apply(outputs)
// outputs = tf.layers.dense({units: 256}).apply(outputs)

let rnnState = []


;[outputs, ...rnnState] = tf.layers.gru({
// outputs = tf.layers.gru({
    units: 8,
    stateful: true,
    returnSequences: true,
    returnState: true
}).apply(outputs)

console.log("RNN inner state:", rnnState.map(s => s.shape))

// outputs = tf.layers.dense({units: 64}).apply(outputs)
// outputs = tf.layers.dense({units: 64}).apply(outputs)
// outputs = tf.layers.dense({units: 64}).apply(outputs)


const model = tf.model({inputs, outputs})

model.compile({optimizer: 'rmsprop', loss: 'meanSquaredError'})

console.log(model.summary())


let busy = false
self.addEventListener('message', async e => {
    const meanRgb = [0.485, 0.456, 0.406]
    const stdRgb = [0.229, 0.224, 0.225]

    const frame = tf.tensor3d(e.data, shape, 'float32')
    const frameNorm = frame
        .div(tf.scalar(255))
        .sub(meanRgb)
        .div(stdRgb)

    // console.log((await frameNorm.array())[0][0])


    if (true && !busy) {
        busy = true

        const res = model.predict(tf.stack([frameNorm]), {
            batchSize
        })
        // res.print()
        console.log("res.shape: ", res.shape)
        // console.log("Pred: ", (await res.array())) //[0][0][0][0]

        const result = model.evaluate(tf.stack([frameNorm]), tf.ones(res.shape), {batchSize});
        result.print();

        const h = await model.fit(tf.stack([frameNorm]), tf.ones(res.shape), {
            batchSize,
            epochs: 1,
            shuffle: false
        })

        console.log("Loss: " + h.history.loss[0])

        busy = false
    }


    const data = await frame.array()
    // console.log(data[0][0])

    frame.dispose()
    frameNorm.dispose()

    self.postMessage(data)
})

