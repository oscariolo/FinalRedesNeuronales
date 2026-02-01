import { ReactNode } from 'react'

export default function TweetSlider({ children }: { children: ReactNode }) {
  return (
    <div className="scroll-box flex flex-col gap-4">
      {children}
    </div>
  );
}