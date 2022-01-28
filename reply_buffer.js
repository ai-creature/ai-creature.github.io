/**
 * Returns a random integer between min (inclusive) and max (inclusive).
 * The value is no lower than min (or the next integer greater than min
 * if min isn't an integer) and no greater than max (or the next integer
 * lower than max if max isn't an integer).
 * Using Math.round() will give you a non-uniform distribution!
 * https://stackoverflow.com/questions/1527803/generating-random-whole-numbers-in-javascript-in-a-specific-range
 */
 const getRandomInt = (min, max)  => {
    min = Math.ceil(min)
    max = Math.floor(max)
    return Math.floor(Math.random() * (max - min + 1)) + min
}

/**
 * Reply Buffer.
 */
class ReplyBuffer {
    /**
     * Constructor.
     * 
     * @param {*} limit maximum number of transitions
     * @param {*} onDiscard callback triggered on discard a transition
     */
    constructor(limit = 500, onDiscard = () => {}) {
        this._limit = limit
        this._onDiscard = onDiscard

        this._buffer = new Array(limit).fill()
        this._pool = []

        this.size = 0
    }

    /**
     * Add a new transition to the reply buffer. 
     * Transition doesn't contain the next state. The next state is derived when sampling.
     * 
     * @param {{id: number, priority: number, state: array, action, reward: number}} transition transition
     */
    add(transition) {
        let { id, priority = 1 } = transition
        if (id === undefined || id < 0 || priority < 1) 
            throw new Error('Invalid arguments')

        const index = id % this._limit

        if (this._buffer[index]) {
            this._pool = this._pool.filter(i => i !== index)
            this._onDiscard(this._buffer[index])
        } else
            this.size++

        while (priority--) 
            this._pool.push(index)

        this._buffer[index] = transition
    }

    /**
     * Return `n` random samples from the buffer. 
     * Returns an empty array if impossible provide required number of samples.
     * 
     * @param {number} [n = 1] - number of samples 
     * @returns array of exactly `n` samples
     */
    sample(n = 1) {
        if (this.size < n) 
            return []

        const 
            sampleIndices = new Set(),
            samples = []

        let counter = n
        while (counter--)
            while (sampleIndices.size < this.size) {
                const randomIndex = this._pool[getRandomInt(0, this._pool.length - 1)]
                if (sampleIndices.has(randomIndex))
                    continue

                sampleIndices.add(randomIndex)

                const { id, state, action, reward } = this._buffer[randomIndex]
                const nextId = id + 1
                const next = this._buffer[nextId % this._limit]

                if (next && next.id === nextId) {
                    samples.push({ state, action, reward, nextState: next.state})
                    break
                }
            }

        return samples.length == n ? samples : []
    }
}

/** TESTS */
(() => {
    return

    const rb = new ReplyBuffer(5)
    rb.add({id: 0, state: 0})
    rb.add({id: 1, state: 1})
    rb.add({id: 2, state: 2, priority: 3})
    
    console.assert(rb.size === 3)
    console.assert(rb._pool.length === 5)
    console.assert(rb._buffer[0].id === 0)
    
    rb.add({id: 2, state: 2})
    rb.add({id: 4, state: 4, priority: 2})
    
    console.assert(rb.size === 4)
    console.assert(rb._pool.length === 5)
    console.assert(JSON.stringify(rb._pool) === '[0,1,2,4,4]')
    
    rb.add({id: 5, state: 0, priority: 2}) // 5%5 = 0 => state = 0
    
    console.assert(rb.size === 4)
    console.assert(rb._pool.length === 6)
    console.assert(rb._buffer.length === 5)
    console.assert(rb._buffer[0].id === 5)
    console.assert(JSON.stringify(rb._pool) === '[1,2,4,4,0,0]')

    console.assert(rb.sample(999).length === 0, 'Too many samples')
    
    let samples1 = rb.sample(2)
    console.assert(samples1.length === 2, 'Only two samples possible')
    console.assert(samples1[0].nextState === (samples1[0].state + 1) % 5, 'Next state should be valid', samples1)

    rb.add({id: 506, state: 506, priority: 3})

    let samples2 = rb.sample(1)
    console.assert(samples2.length === 1, 'Only one suitable sample with valid next state')
    console.assert(samples2[0].state === 4, 'Sample with id:4')
    console.assert(rb._buffer[1].id === 506, '506 % 5 = 1')

    console.assert(rb.sample(2).length === 0, 
        'Can not sample 2 transitions since next state is available only for one state')
})()
