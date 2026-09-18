import sys
import uuid
from langchain_core.messages import HumanMessage
from app.agent.workflow import biodiversity_agent


def main():
    print("=" * 60)
    print("Biodiversity AI Agent CLI")
    print("Type 'exit' to quit | Type 'clear' to start a new topic thread.")
    print("=" * 60)

    session_thread_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting CLI...")
                break

            if user_input.lower() in ["clear", "reset"]:
                session_thread_id = str(uuid.uuid4())
                print("Session memory cleared. Topic thread reset.")
                continue

            config = {"configurable": {"thread_id": session_thread_id}}

            input_payload = {
                "user_query": user_input,
                "user_prompt": user_input,
                "messages": [HumanMessage(content=user_input)],
            }

            result = biodiversity_agent.invoke(input_payload, config=config)

            print("\nAgent:\n")
            response_text = result.get("final_response") or ""
            if not response_text and result.get("messages"):
                response_text = result["messages"][-1].content

            print(response_text)

        except KeyboardInterrupt:
            print("\nSession ended.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[Error Details]: {type(e).__name__} - {e}")


if __name__ == "__main__":
    main()
