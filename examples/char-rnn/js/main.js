const fs = require('fs')

const SEQLEN = 40
const SKIP = 3
const EPOCHS = 10
const LEARNING_RATE = 0.01
const BATCH_SIZE = 128
const LOADMODEL = false

function loadFile(path) {
    return new Promise((resolve, reject) => {
        fs.readFile(path, (err, data) => {
            if (err) reject(err)
            else resolve(data)
        })
    })
}

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

async function getData(path) {
    
    const buf = await loadFile(path)
    const text = buf.toString().toLowerCase().slice(0, 250000)

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
    
    // one-hotify
    // const x = tf.zeros([sentences.length, SEQLEN, chars.length], 'float32')
    // const y = tf.zeros([sentences.length, chars.length], 'float32')
    const xBuff = tf.buffer([sentences.length, SEQLEN, chars.length])
    const yBuff = tf.buffer([sentences.length, chars.length])
    sentences.forEach((sentence, i) => {
        sentence.split('').forEach((char, t) => {
            // console.log(char)
            // console.log(i, t, charIndicies[char], char)
            xBuff.set(1, i, t, charIndicies[char])
        })
        yBuff.set(1, i, charIndicies[nextChars[i]])
    })
    
    const x = xBuff.toTensor()
    const y = yBuff.toTensor()
    
    return {
        x, y, charIndicies, chars, numBatches: Math.floor(sentences.length / BATCH_SIZE)
    }
}

function sample(preds) {
    // console.log(preds)
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

        const preds = model.predict(x, { verbose: true }).dataSync()
        const y = sample(preds)
        const char = chars[y]
        
        seed.shift(); seed.push(char)
        output.push(char)
        
        x.dispose()
        await tf.nextFrame()
    }
    
    return output.join('')
}

async function main() {
    
    const data = await getData(`${__dirname}/data/tinyshakespeare.txt`)
    let model = null

    const seed = "This is a test sentence. It will be used to seed the model."
    
    if (LOADMODEL) {
        console.log('Loading model from IndexedDB')
        model = await tf.loadModel('indexeddb://model')
    } else {
        console.log('Training model...')
        model = getModel([SEQLEN, data.chars.length])
        let history = null
        for (let i = 0; i < EPOCHS; i++) {
            const then = Date.now()
            for (let batch = 0; batch < data.numBatches; batch++) {
                const batchX = tf.slice(data.x, batch * BATCH_SIZE, BATCH_SIZE)
                const batchY = tf.slice(data.y, batch * BATCH_SIZE, BATCH_SIZE)
                history = await model.fit(batchX, batchY, { batchSize: BATCH_SIZE }) 
                batchX.dispose()
                batchY.dispose()
                await tf.nextFrame()
            }
            console.log(`Epoch ${i + 1} loss: ${history.history.loss[0]}, accuracy: ${history.history.acc[0]}`)
            console.log(`Epoch lasted ${((Date.now() - then) / 1000).toFixed(0)} seconds`)            
            // const text = await generate(seed, 50, data.charIndicies, data.chars, model)
            // console.log(text)
            await model.save('indexeddb://model')
            const text = await generate(seed, 100, data.charIndicies, data.chars, model)
            console.log(text)
        }
    }
    
    console.log(`Finished training for ${EPOCHS}`)
    const text = await generate(seed, 100, data.charIndicies, data.chars, model)
    console.log(text)
    
    data.x.dispose()
    data.y.dispose()
}

main()