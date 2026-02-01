const API_URL = 'http://localhost:8000';

export interface Tweet{
    id: string;
    text: string;
    user: {
        id: string;
        name: string;
        username: string;
    };
    sentiment: 'positive' | 'neutral' | 'negative';
}

export async function fetchLastTweets(): Promise<Tweet[]> {
    // For demo purposes, return empty array initially
    // In production, you might fetch from a database endpoint
    return Promise.resolve([]);
}

export async function getModels(): Promise<string[]> {
    try {
        const response = await fetch(`${API_URL}/models`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Fetched models:', data);
        return data.models;
    } catch (error) {
        console.error('Error fetching models:', error);
        return [];
    }
}


export async function postTweet(text: string, model: string): Promise<Tweet> {
    try {
        const response = await fetch(`${API_URL}/tweet/${model}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                user: {
                    id: "0987654321",
                    name: "John Doe",
                    username: "johndoe"
                }
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const tweet = await response.json();
        return tweet;
    } catch (error) {
        console.error('Error posting tweet:', error);
        // Fallback to simulated response if server is not available
        return {
            id: Date.now().toString(),
            text: text,
            user: {
                id: "0987654321",
                name: "John Doe",
                username: "johndoe"
            },
            sentiment: 'neutral'
        };
    }
}

