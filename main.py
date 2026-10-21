from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# Hotel rooms
rooms = {
    101: {"type": "Single", "capacity": 1, "price": 5000},
    102: {"type": "Single", "capacity": 1, "price": 5000},
}

# Save Bookings 
bookings = []


# Check the date format
def valid_date(date: str) -> bool:
    if len(date) != 10:
        return False
    if date[4] != "-" or date[7] != "-":
        return False
    numbers = date[:4] + date[5:7] + date[8:]
    return numbers.isdigit()

# check availability
@tool
def check_rooms(date: str = "") -> str:
    """Show hotel room types, guest capacity, prices and availability for a date (YYYY-MM-DD). Leave date empty to list rooms and prices only."""
    if date != "" and not valid_date(date):
        return "Please enter the date in YYYY-MM-DD format."

    answer = "Hotel rooms:\n"
    for room_number in rooms:
        room = rooms[room_number]
        line = f"Room {room_number}: {room['type']}, {room['capacity']} guest(s), Rs. {room['price']} per booking date"

        if date != "":
            status = "Available"
            for booking in bookings:
                if booking["room"] == room_number and booking["date"] == date:
                    status = "Booked"
            line = line + ", " + status

        answer = answer + line + "\n"

    if date == "":
        answer = answer + "Give a booking date to check availability."
    return answer


#book room
@tool
def book_room(name: str, room_number: int, guests: int, date: str) -> str:
    """Book a hotel room for one whole date (YYYY-MM-DD). Needs guest name, room number, number of guests and date."""
    name = name.strip()
    if name == "":
        return "Please provide the guest's name."
    if room_number not in rooms:
        return "This room does not exist."
    if not valid_date(date):
        return "Please enter the booking date in YYYY-MM-DD format."
    if guests < 1 or guests > rooms[room_number]["capacity"]:
        return "Invalid guest count for this room."

    for booking in bookings:
        if booking["room"] == room_number and booking["date"] == date:
            return "Room is already booked for this date. Choose another room or date."

    bookings.append({
        "name": name,
        "room": room_number,
        "guests": guests,
        "date": date,
    })
    price = rooms[room_number]["price"]
    return f"BOOKING CONFIRMED: {name}, Room {room_number}, {guests} guest(s), {date}. Price: Rs. {price}."


llm = ChatOllama(model="llama3.2", temperature=0)

agent = create_agent(
    llm,
    [check_rooms, book_room],
    system_prompt=(
        "You are the Green Palace Hotel assistant. Answer in simple English. "
        "Use check_rooms for room types, prices and availability. "
        "Use book_room for bookings. "
        "Each booking is for one whole calendar date, not a night or a date range. "
        "Dates must be YYYY-MM-DD. Ask for any missing booking information. "
        "Never say a booking is confirmed unless the tool says so. "
        "If a question is about hotel facilities or policies, say the information is not available yet."
    )
)

#run
def main():
    print("------------------------------------")
    print("   GREEN PALACE HOTEL AI ASSISTANT")
    print("------------------------------------")
    print("AI Assistant Started!\n")

    history = []
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if question == "":
            continue

        history.append(("user", question))
        previous_messages = len(history)

        try:
            result = agent.invoke({"messages": history}, {"recursion_limit": 15})
        except Exception as e:
            print("Error:", e)
            history.pop()
            continue

        # Print answers
        print("AI:", result["messages"][-1].content, "\n")
        history = result["messages"]

