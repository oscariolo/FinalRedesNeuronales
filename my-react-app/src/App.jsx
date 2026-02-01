import { useState, useEffect } from 'react'
import './App.css'
import { fetchLastTweets, postTweet } from './services/twitterServices'
import TweetComponent from './components/TweetComponent'
import TweetSlider from './components/TweetSlider'

function App(){
  const [tweets, setTweets] = useState([])
  const [newTweet, setNewTweet] = useState('')
  const [isPosting, setIsPosting] = useState(false)
  
  useEffect(() => {
    const loadTweets = async () => {
      const data = await fetchLastTweets()
      setTweets(data)
    }
    loadTweets()
  }, [])

  const handleSubmit = async (event) => {
    event.preventDefault()
    const text = newTweet.trim()
    if (!text) return

    setIsPosting(true)
    try {
      const created = await postTweet(text)
      setTweets((prev) => [created, ...prev])
      setNewTweet('')
    } finally {
      setIsPosting(false)
    }
  }

  return (
    <>
      <section className="mb-6 mx-auto w-full max-w-2xl rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
        <h1 className="text-2xl font-semibold text-white">Sentiment Tweets</h1>
        <p className="text-sm text-slate-400">Share what’s happening.</p>
        <form onSubmit={handleSubmit} className="mt-4 flex flex-col gap-3">
          <textarea
            className="min-h-27.5 w-full resize-none rounded-xl border border-slate-800 bg-slate-950 p-3 text-sm text-slate-100 outline-none focus:ring-2 focus:ring-cyan-500"
            placeholder="What’s happening?"
            value={newTweet}
            onChange={(event) => setNewTweet(event.target.value)}
          />
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>{newTweet.trim().length} / 280</span>
            <button
              type="submit"
              disabled={isPosting || newTweet.trim().length === 0}
              className="rounded-full text-amber-50 bg-cyan-500 px-4 py-2 text-sm font-semiboldtransition disabled:cursor-not-allowed disabled:bg-slate-700"
            >
              {isPosting ? 'Posting...' : 'Tweet'}
            </button>
          </div>
        </form>
      </section>
      
      <div className="mx-auto w-full max-w-2xl">
        <TweetSlider>
          {tweets.map((tweet) => (
            <TweetComponent key={tweet.id} tweet={tweet} />
          ))}
        </TweetSlider>
      </div>
    </>
  )
}

export default App
