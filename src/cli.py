from chatbot import chat

print("\nMental Health Support Chatbot\n(type exit to quit)\n")

while True:
    msg = input("You: ")
    if msg.lower() == "exit":
        break
    print("Bot:", chat(msg))
