import openai, os

import rospy
import actionlib
import random
import speech_recognition as sr
import time

from tmc_msgs.msg import Voice
from std_msgs.msg import String
from mas_execution_manager.scenario_state_base import ScenarioStateBase
from hsrb_interface import Robot

openai.api_key = os.getenv("OPENAI_API_KEY")

initial_prompt = [{"role":"user", "content":"You are Lucy, a Toyota HSR robot that is a part of b it bots team. \
                   You are friendly and helpful, and you are always ready to help people. \
                   You love to talk to people and learn about them. \
                   You do not have the abilitz to browse the internet or make calls, so do not offer to do so. \
                   You should also refer to yourself as a robot and not as a AI language model always.\
                   You can do a lot of things, such as navigating to a location, grasping objects, and recognizing objects. \
                   But to do so, they need to say the specific commands. Try to speak in small sentences."}]

conversation_history = []
max_answer_duration_s = 30
max_api_calls_per_answer = 5


class LucyChatbot:
    def __init__(self):
        self.r = sr.Recognizer()
        #self.mic = sr.Microphone(device_index=9) # For RODE, SAMSON microphones.
        self.mic = sr.Microphone() # For HSR's default microphone.

        with self.mic as source:
                self.r.adjust_for_ambient_noise(source)


        #self.say_pub = rospy.Publisher('/talk_request', Voice, latch=True, queue_size=1)

    #def say(self, sentence):
        #say_msg = String()
        #say_msg.data = sentence
        #self.say_pub.publish(say_msg)

    def listen_to_audio(self):
        try:
            with self.mic as source:
                print("Please speak now!")
                audio = self.r.listen(source)
                #audio = self.r.record(source,duration=6)
                try:
                    user_input = self.r.recognize_google(audio)
                    user_input = user_input.lower()
                    print("You said:", user_input)
                    return user_input
                except:
                    user_input = self.r.recognize_sphinx(audio)
                    user_input = user_input.lower()
                    print("You said:", user_input)
                    return user_input
        except sr.UnknownValueError:
            print(self.generate_error_response())
            return None
        except sr.RequestError as e:
            print("Sorry, there was an error processing your request. Please try again later.")
            return None
        except Exception as e:
            print(f"Sorry, something went wrong: {e}")
            return None

def calculate_speech_duration(text_length):
    # Assuming an average speaking rate of 150 words per minute
    words_per_minute = 150
    seconds_per_word = 60 / words_per_minute
    speech_duration = text_length * seconds_per_word
    return speech_duration
    

def ask(question):
    global conversation_history
    prompt = initial_prompt + conversation_history + [{"role":"user", "content":question}]
    speech_duration = 1e15
    api_calls = 0
    while speech_duration > max_answer_duration_s and \
          api_calls < max_api_calls_per_answer:
        response = openai.ChatCompletion.create(
                    model= "gpt-3.5-turbo",
                    messages=prompt,
                    max_tokens=1000,
                    n=1,
                    temperature=0.9
                )
        answer = response['choices'][0]['message']['content']
        conversation_history.append({"role":"user", "content":question})
        conversation_history.append({"role":"assistant", "content":answer})
        speech_duration = calculate_speech_duration(len(answer.split()))
        print(f"Generated answer with estimated duration of {speech_duration} seconds")
        prompt += [{"role":"user", "content": "Shorten the answer."}]
        api_calls += 1
    return answer

if __name__ == "__main__":
    #rospy.init_node('lucy_chatbot')
    have_conversation_flag = False
    robot = Robot()
    mybot = LucyChatbot()
    tts = robot.try_get('default_tts')
    tts.language = tts.ENGLISH
    print("Chatbot is ready")
    while True:
        question = mybot.listen_to_audio()
        if question is not None:
            question = question.lower()
            if "start" in question and "lucy" in question :
                have_conversation_flag = True
            if "stop" in question and "lucy" in question:
                have_conversation_flag = False
            if have_conversation_flag:
                ans=ask(question)
                #print("###########################")
                #print(ans)
                #print("###########################")
                speech_duration = calculate_speech_duration(len(ans.split()))
                print(speech_duration)
                tts.say(ans)
                time.sleep(speech_duration)
        else:
            continue
