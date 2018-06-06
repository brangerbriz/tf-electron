const SEQLEN = 40
const SKIP = 3
const EPOCHS = 10
const LEARNING_RATE = 0.01
const BATCH_SIZE = 128
const LOADMODEL = false

// uncomment this if you want to use IPC to communicate with the parent page
// const { ipcRenderer } = require('electron')

// promisified fs.readFile()
function loadFile(path) {
    const fs = require('fs')
    return new Promise((resolve, reject) => {
        fs.readFile(path, (err, data) => {
            if (err) reject(err)
            else resolve(data)
        })
    })
}

// define and return an LSTM RNN model architecture. RNNs are used
// with sequential data.
function getModel(inputShape) {
    const model = tf.sequential()
    model.add(tf.layers.lstm({units: 128, inputShape: inputShape }))
    model.add(tf.layers.dense({units: inputShape[1]}))
    model.add(tf.layers.activation({ activation: 'softmax' }))
    model.compile({
        optimizer: tf.train.rmsprop(LEARNING_RATE),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })
    return model
}

// utility function for loading and prepairing training data. Returns an object.
async function getData(path) {
    
    const buf = await loadFile(path)
    // convert all of the characters to lowercase to reduce our number of 
    // output classes
    const text = buf.toString().toLowerCase()
    console.log(`corpus length: ${text.length}`)    
    
    const chars = Array.from(new Set(text))
    console.log(`total chars: ${chars.length}`)
    
    const charIndicies = {}; chars.forEach((c, i) => charIndicies[c] = i)
    
    // cut the text in semi-redundant sequences of maxlen characters
    const sentences = []
    const nextChars = []
    for (let i = 0; i < text.length - SEQLEN; i += SKIP) {
        sentences.push(text.slice(i, i + SEQLEN))
        nextChars.push(text.slice(i + SEQLEN, i + SEQLEN + 1))
        // console.log(text.slice(i, i + SEQLEN), '->', text.slice(i + SEQLEN, i + SEQLEN + 1))
    }
    
    // each array element will hold a batch of BATCH_SIZE examples
    const X = []
    const Y = []
    const partitionSize = BATCH_SIZE * 100
    for (let batch = 0; batch < sentences.length; batch += partitionSize) {

        // we can't store all of the data in GPU memory, so instead we use
        // tf.TensorBuffer objects which are allocated by the CPU. Becore we
        // train on a batch we transform it into a Tensor using TensorBuffer.toTensor()
        
        const xBuff = tf.buffer([partitionSize, SEQLEN, chars.length])
        const yBuff = tf.buffer([partitionSize, chars.length])
        
        const sentenceBatch = sentences.slice(batch, batch + partitionSize)
        const nextCharsBatch = nextChars.slice(batch, batch + partitionSize)
        
        // one-hotify batches
        sentenceBatch.forEach((sentence, j) => {
            sentence.split('').forEach((char, k) => {
                xBuff.set(1, j, k, charIndicies[char])
            })
            yBuff.set(1, j, charIndicies[nextCharsBatch[j]])
        })

        X.push(xBuff)
        Y.push(yBuff)
    }
    return {
        X, Y, charIndicies, chars
    }
}

// sample an output from the multinomial distribution output of model.predict()
function sample(preds) {
    let probas = Sampling.Multinomial(1, preds).draw()
    // lazy argmax that returns the index of the "hot" value in a one-hot vector 
    return probas.reduce((acc, val, i) => acc + (val == 1 ? i : 0))
}

async function generate(seed, numChars, charIndicies, chars, model) {
    console.assert(seed.length >= SEQLEN)
    seed = seed.toLowerCase().split('').slice(0, SEQLEN)
    const output = []
    for (let i = 0; i < numChars; i++) {
        
        const x = tf.zeros([1, SEQLEN, chars.length])
        seed.forEach((char, j) => {
            x.buffer().set(1, 0, j, charIndicies[char])
        })

        const preds = model.predict(x, { verbose: true })
        const y = sample(preds.dataSync())
        const char = chars[y]
        
        seed.shift(); seed.push(char)
        output.push(char)
        
        preds.dispose()
        x.dispose()
    }
    
    return output.join('')
}

async function main() {
    const data = await getData(`${__dirname}/data/tinyshakespeare.txt`)
    let model = null
    const seed = "This is a test sentence. It will be used to seed the model."
    
    // load model from IndexedDB if the flag says so
    if (LOADMODEL) {
        console.log('Loading model from IndexedDB')
        model = await tf.loadModel('indexeddb://model')
    } else {
        // otherwise train a new model and save it to IndexedDB overwriting any
        // existing models that may be saved there
        console.log('Training model...')
        model = getModel([SEQLEN, data.chars.length])
        let history = null
        for (let i = 0; i < EPOCHS; i++) {
            const then = Date.now()
            for (let batch = 0; batch < data.X.length; batch++) {
                const batchX = data.X[batch].toTensor()
                const batchY = data.Y[batch].toTensor()
                history = await model.fit(batchX, batchY, { batchSize: BATCH_SIZE }) 
                batchX.dispose()
                batchY.dispose()
            }
            console.log(`Epoch ${i + 1} loss: ${history.history.loss[0]}, accuracy: ${history.history.acc[0]}`)
            console.log(`Epoch lasted ${((Date.now() - then) / 1000).toFixed(0)} seconds`)
            await model.save('indexeddb://model')
            const text = await generate(seed, 100, data.charIndicies, data.chars, model)
            console.log(text)
        }
    }
    
    console.log(`Finished training for ${EPOCHS}`)
    console.log(`Generating 1000 characters of synthetic text:`)
    const text = await generate(seed, 1000, data.charIndicies, data.chars, model)
    console.log(text)
    
}

main()