import speech_recognition as sr
import pyaudio
import pyttsx3
import pyautogui as pg


def listen():
    r = sr.Recognizer()
    microphone = sr.Microphone()        

    r.energy_threshold = 600
    r.dynamic_energy_threshold = False
    

    word_map = {'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8',
                'nine':'9','zero':'0','question':'?','exclamation':'!','percentage':'%','dollar':'$',
                'comma':',','dot':'.','asterisk':'*','plus':'+','minus':'-','equals':'=','apostrophe':"'"}

    commmand_map = { 'alt left':'altleft','alt right':'altright','caps lock':'capslock','control':'ctrl','control right':'ctrlright',
                'control left':'ctrlleft','print screen':'printscreen','shift left':'shiftleft','shift right':'shiftright',
                'page down':'pagedown','page up':'pageup','volume down':'volumedown', 'volume mute':'volumemute', 'volume up':'volumeup',
                'space bar':'space','print screen':'printscreen','open round bracket':'(','close round bracket':')'}

    commands =  ['alt', 'altleft', 'altright', 'backspace',
                'browserback', 'browserfavorites', 'browserforward', 'browserhome',
                'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear',
                'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete', 'down', 'end', 'enter', 'esc', 'escape', 'f1', 'f10',
                'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20',
                'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
                'final', 'insert', 'left', 'multiply', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6',
                'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn',
                'pgup', 'playpause', 'prevtrack', 'printscreen', 'prntscrn',
                'prtsc', 'prtscr', 'right', 'scrolllock', 'select', 'separator',
                'shift', 'shiftleft', 'shiftright', 'tab',
                'up', 'volumedown', 'volumemute', 'volumeup']

    ignore = 0  #Parameter to indicate if text needs to be written or not
    start = 0;
    while(True):    
        try:
            with microphone as source2:
                if start is 0:
                    r.adjust_for_ambient_noise(source2,duration=1)
                    start = 1

                r.adjust_for_ambient_noise(source2,duration=0.1)
                #Represents the minimum length of silence (in seconds) that will register as the end of a phrase
                r.pause_threshold = 0.9
                #To notify speaker to start to speak
                print("Speak")
                
                audio2 = r.listen(source2,phrase_time_limit=20,timeout=10)
                text = r.recognize_google(audio2, language="en-IN") 
                text = text.lower()
                words = text.split()
                
                for i in range(len(words)):
                    words[i] = word_map.get(words[i],words[i])
                                
                final_text = " ".join(words)              #Text we got after processing
                
                print(final_text)

                if final_text == 'pause':
                    ignore = 1
                elif final_text == "continue":
                    ignore = 0
                elif not ignore:
                    if final_text in commands or final_text in commmand_map.keys():
                        pg.press(commmand_map.get(final_text,final_text))
                    else:
                        pg.write(final_text,0.2)
                    print(final_text)
                elif final_text == "stop":
                    #Notify speaker program has terminated
                    print("Stopped")
                    break

                    ##r.adjust_for_ambient_noise(source2,duration=1)        

        except sr.RequestError as e: 
            print("Could not request results; {0}".format(e)) 
                  
        except sr.UnknownValueError: 
            print("unknown error occured")
        except sr.WaitTimeoutError:
            print("error")

            
        
listen()