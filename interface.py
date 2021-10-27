import threading
import shutil
import PySimpleGUI as sg
from gen_mask import ocr_img_mask
from inpaint import lama_inpainting

def the_gui():
    """
    Starts and executes the GUI
    Reads data from a global variable and displays
    Returns when the user exits / closes the window
    """
    sg.theme('Dark Blue 3')

    layout = [[sg.Text('Remove Text')], 
            [sg.Text('SOURCE FOLDER', size=(15, 1)), sg.InputText(key='-SOURCE FOLDER-'), sg.FolderBrowse(target='-SOURCE FOLDER-')],
            [sg.Text('TARGET FOLDER', size=(15, 1)), sg.InputText(key='-TARGET FOLDER-'), sg.FolderBrowse(target='-TARGET FOLDER-')],
            [sg.Button('Start')], 
            [sg.Text('Work progress:'), sg.ProgressBar(100, size=(33, 15), orientation='h', key='-PROG-'), sg.Text('',size=(7, 1), key='-FINISH-')]]
                
    # Create the window
    window = sg.Window('Remove Text', layout)

    thread = None
    tmp_dir = None

    # --------------------- EVENT LOOP ---------------------
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'Start':
            #print(event, values)
            source_path, target_path = values['-SOURCE FOLDER-'], values['-TARGET FOLDER-']
            #print(source_path)
            #print(target_path)
            thread = threading.Thread(target=ocr_img_mask, args=(source_path, window), daemon=True)
            thread.start()
        elif event == '-PROGRESS-':
            window['-PROG-'].update_bar(values[event], 100)
        elif event == '-OCR THREAD-':
            thread.join(timeout=0)
            tmp_dir = values[event]
            thread = threading.Thread(target=lama_inpainting, args=(tmp_dir, target_path, window), daemon=True)
            thread.start()
        elif event == '-LAMA THREAD-':
            thread.join(timeout=0)
            shutil.rmtree(tmp_dir)
            print("Remove Text Done!")
            window['-FINISH-'].update('FINISH!')

    # Finish up by removing from the screen
    window.close()

if __name__ == '__main__':
    the_gui()
    print('Exiting Program.')