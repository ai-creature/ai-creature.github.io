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

const model = tf.sequential();
const inputShape = [128, 256, 3]
const kernelSize = [3, 3]
const poolSize = 2
const strides = 1
const padding = 'same'
let filterPow = 3


model.add(tf.layers.inputLayer({
    inputShape, 
    batchSize : 1
}))

layers = 14
for (let i = 0; i < layers; i++) {
    if (i%3 == 1) {
        filterPow++
        
    }

    model.add(tf.layers.conv2d({
        filters: 2**filterPow,
        kernelSize,
        activation: 'relu',
        padding,
        strides,
        kernelInitializer: 'heNormal'
    }))

    if (i%2 == 1) {
        model.add(tf.layers.maxPooling2d({
            poolSize
        }))
    }
}

// const reshapeLayer = tf.layers.convLstm2d()


model.add(tf.layers.reshape({targetShape: [1, 512]}))

model.add(tf.layers.dense({units: 256}))
model.add(tf.layers.dense({units: 256}))
model.add(tf.layers.dense({units: 256}))

model.add(tf.layers.rnn({
    cell: [
        tf.layers.lstmCell({units: 128}),
        tf.layers.lstmCell({units: 128}),
        tf.layers.lstmCell({units: 128})
    ], 
    returnSequences: true,
    returnState: false,
    stateful: true
}))

model.add(tf.layers.dense({units: 64}))
model.add(tf.layers.dense({units: 64}))
model.add(tf.layers.dense({units: 64}))



model.compile({optimizer: 'adam', loss: 'meanSquaredError'})
console.log(model.summary())




let busy = false
self.addEventListener('message', async e => {
    const meanRgb = [0.485, 0.456, 0.406]
    const stdRgb = [0.229, 0.224, 0.225]

    const frame = tf.tensor3d(e.data, inputShape, 'float32')
    const frameNorm = frame
        .div(tf.scalar(255))
        .sub(meanRgb)
        .div(stdRgb)

    // console.log((await frameNorm.array())[0][0])



    if (true && !busy) {
        busy = true

        console.log("getBackend: " + tf.getBackend ())
        let st = Date.now()
        // console.time("predict: " + st)
        const res = model.predict(frameNorm.expandDims(0))
        // res.print()
        // console.log(res.shape)
        // console.log("Pred: ", (await res.array())) //[0][0][0][0]
        // return
        // console.timeEnd("predict: " + st)



        // st = Date.now()
        // console.time("fit: " + st)
        const h = await model.fit(frameNorm.expandDims(0), tf.ones(res.shape), {
            batchSize: 1,
            epochs: 1,
            shuffle: false
        });
        console.log("Loss: " + h.history.loss[0])
        // console.timeEnd("fit: " + st)

        busy = false
    }


    const data = await frame.array()
    // console.log(data[0][0])

    frame.dispose()
    frameNorm.dispose()

    self.postMessage(data)
})

