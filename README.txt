How to set up and run the project:

1. Install dependencies:
   - on "...\MemoWeave\backend", DO: pip install -r requirements.txt
   - on "...\MemoWeave\frontend", DO:  npm install

2. Run the frontend:
   - Open powershell instance
   - You should be on "...\MemoWeave\frontend"
   - Do npm run dev

3. Run the server:
   - Remain on root folder ("...\MemoWeave")
   - Here, DO: uvicorn server:app --host 0.0.0.0 --port 8000 (important; running on different ports will cause unexpected errors)

** You can split one terminal instance to two so that you can monitor both server-run and frontend-run at the same time. **