from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get('/')
def read_main():
    return {"message": "Hello World of FastAPI HTTPS"}


if __name__ == '__main__':
    uvicorn.run("FastApiMain:app",
                host="0.0.0.0",
                port=8432,
                reload=True,
                ssl_keyfile="/home/amitli/Repo/Voice/VoiceApp/key.pem",
                ssl_certfile="/home/amitli/Repo/Voice/VoiceApp/cert.pem"
                )

