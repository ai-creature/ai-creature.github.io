(async () => {
    importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js");
    importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js");
    
                    
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // Generate some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7, 8], [5, 1]);

    // Train the model using the data.
    // await model.fit(xs, ys, {epochs: 100});

    // Use the model to do inference on a data point the model hasn't seen before:
    // model.predict(tf.tensor2d([6], [1, 1])).print();

    self.addEventListener('message', async (e) => {

        // const ta = new Int32Array(e.data)
        // console.log(new Uint8Array(e.data))
        // const frame = tf.tensor3d(e.data, [200,200,3])

        // var blob = new Blob( [ e.data ], { type: "image/png" } );

        // const frame = tf.browser.fromPixels(blob)

        // console.log(`Worker: `, e.data.shape)

        const frame = tf.tensor3d(e.data, [75, 150, 3])

        // const base64Data = e.data
        // const DecodeBase64ToBinary = (base64Data) => {
        //     const decodedString = atob(base64Data);
        //     const bufferLength = decodedString.length;
        //     const bufferView = new Uint8Array(new ArrayBuffer(bufferLength));

        //     for (let i = 0; i < bufferLength; i++) {
        //         bufferView[i] = decodedString.charCodeAt(i);
        //     }

        //     return bufferView;
        // };

        // const frame = tf.tensor3d(dsd, [100, 200, 3])
        
        // const frame = tf.node.decodeJpeg(DecodeBase64ToBinary(base64Data))
        const data = await frame.array()
        self.postMessage({
            data,
            console: "" 
        })
    })
})()