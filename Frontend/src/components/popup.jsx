import { useState } from "react";

export default function Popup() {
    const [videoId, setVideoId] = useState("");
    const [query, setQuery] = useState("");
    const [response, setResponse] = useState("");
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setResponse("");

        try {
            const res = await fetch("http://127.0.0.1:8000/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ video_id: videoId, query }),
            });

            if (!res.ok) throw new Error(`Server returned ${res.status}`);
            const data = await res.json();
            setResponse(data.answer || JSON.stringify(data));
        } catch (err) {
            setResponse("Error: " + err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-[500px] h-[400px] bg-gray-50 p-4">
            <div className="bg-white rounded-lg shadow p-4 flex flex-col gap-4 h-full">

                {/* Row 1: Title + Video ID */}
                <div className="flex justify-between items-center">
                    <h1 className="text-lg font-bold">🎥 YouTube Chatbot</h1>
                    <input
                        type="text"
                        placeholder="Video ID"
                        value={videoId}
                        onChange={(e) => setVideoId(e.target.value)}
                        className="border border-gray-300 rounded px-2 py-1 text-sm w-[200px] focus:ring-2 focus:ring-blue-400 outline-none"
                    />
                </div>

                {/* Row 2: Ask input */}
                <form onSubmit={handleSubmit} className="flex flex-col gap-2">
                    <textarea
                        placeholder="Ask something about the video..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        className="border border-gray-300 rounded px-3 py-2 text-sm focus:ring-2 focus:ring-blue-400 outline-none resize-none w-full"
                        rows={3}
                    />
                    <button
                        type="submit"
                        disabled={loading}
                        className="bg-blue-500 text-white py-2 rounded hover:bg-blue-600 transition disabled:opacity-60 w-full"
                    >
                        {loading ? "Thinking..." : "Ask"}
                    </button>
                </form>

                {/* Row 3: Response */}
                <div className="flex-1 overflow-y-auto bg-gray-50 border border-gray-300 rounded p-2 text-sm leading-snug whitespace-pre-wrap w-full">
                    {response || "No response yet..."}
                </div>
            </div>
        </div>
    );
}
