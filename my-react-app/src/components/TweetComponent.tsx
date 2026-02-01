import { Tweet } from "../services/twitterServices";

export default function TweetComponent({ tweet }: { tweet: Tweet }) {
  const sentimentColors = {
    positive: 'bg-green-500/20 border-green-500/50',
    neutral: 'bg-slate-800 border-slate-700',
    negative: 'bg-red-500/20 border-red-500/50'
  };

  const sentimentEmoji = {
    very_positive: '😃',
    positive: '😊',
    neutral: '😐',
    negative: '😞'
  };

  return (
    <article className={`min-w-100 max-w-[320px] shrink-0 rounded-2xl border p-4 text-left self-center ${sentimentColors[tweet.sentiment]}`}>
      <div className="flex items-center justify-between mb-2">
        <header className="text-sm text-slate-400">@{tweet.user.username}</header>
        <span className="text-lg" title={tweet.sentiment}>{sentimentEmoji[tweet.sentiment]}</span>
      </div>
      <p className="mt-2 text-sm text-slate-100">{tweet.text}</p>
    </article>
  );
}