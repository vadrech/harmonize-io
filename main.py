def MidiToCsv(file):
    import py_midicsv as pm
    import pandas as pd

    csv_file = pm.midi_to_csv(file)
    
    dataframe = pd.DataFrame([sub.split(",") for sub in csv_file])

    del csv_file, file

    return dataframe
 
    
def ReFormatDataFrame(melody_original, melody_velocities):
    import pandas as pd

    description_column = melody_original.iloc[:, 2]
    
    tempo_row, time_signature_row = "", ""
    
    for row_number, row in enumerate(description_column):
        if "tempo" in row.lower():
            tempo_row = row_number
            
        elif "time_signature" in row.lower():
            time_signature_row = row_number
            
        if tempo_row != "" and time_signature_row != "":
            break
            
            
    tempo_value = int(melody_original.values[row_number][3])
    time_signature_value = [int(melody_original.values[time_signature_row][3]), int(melody_original.values[time_signature_row][4])]

    melody_original = pd.DataFrame(melody_original.iloc[:, [1, 4, 5]].values).dropna()

    melody_original = pd.DataFrame([row for row in melody_original.values if int(row[0]) != 0])
    
    melody_original = pd.DataFrame([[int(a), int(b)] for (a, b, c) in melody_original.values])
    
    melody_original = pd.DataFrame([[0, melody_original.values[0][1], melody_velocities]]).append(melody_original)

    memory_of_notes = []
    melody_original_list = []
    
    for row in melody_original.values:
        if row[1] in memory_of_notes:
            memory_of_notes.remove(row[1])
            melody_original_list.append([row[0], row[1], 0])
            
        else:
            memory_of_notes.append(row[1])
            melody_original_list.append([row[0], row[1], melody_velocities])
    
    melody_original = pd.DataFrame(melody_original_list)
    
        
    melody = melody_original.iloc[:, [0, 1]]
    
    melody_minimum = int(melody.iloc[:, 1].min())

    del melody_velocities, description_column, melody_original_list, row, row_number, tempo_row, time_signature_row

    return melody, melody_original, melody_minimum, tempo_value, time_signature_value


def LockingToGrid1(melody, harmonies_per_bar, time_signature_value):
    import pandas as pd
        
    cut_melody = []
    memory_of_notes = []
    
    for row in melody.values:
        if row[1] in memory_of_notes:
            memory_of_notes.remove(row[1])
            
        else:
            memory_of_notes.append(row[1])
            cut_melody.append(row)
    
    cut_melody = pd.DataFrame(cut_melody)
    
    melody = pd.concat((cut_melody.iloc[:, 0] * harmonies_per_bar // int(((time_signature_value[0]) / (2 ** time_signature_value[1])) * 1920), cut_melody.iloc[:, 1]), axis = 1)

    melody = pd.concat((melody.iloc[:, 0] + 1, melody.iloc[:, 1]), axis = 1)
    
    del cut_melody, memory_of_notes, row, time_signature_value
    
    return melody


def LockingToGrid2(melody, number_of_melody_notes_to_consider):
    import pandas as pd
    
    bar_numbers = [int(melody.values[0][0])]
    melody_notes_in_bar = []

    melody_grouped_by_period = []

    for line_number, note in enumerate(melody.values):
        bar_number = int(note[0])
        melody_note = int(note[1])
        
        if bar_number in bar_numbers:
            melody_notes_in_bar.append(melody_note)
            
        else:
            melody_grouped_by_period.append([bar_numbers[-1], melody_notes_in_bar])
            melody_notes_in_bar = [melody_note]
            bar_numbers.append(bar_number)
            
        if line_number == (len(melody.values) - 1):
            melody_grouped_by_period.append([bar_numbers[-1], melody_notes_in_bar])
            
            
    important_melody_by_period = []
    
    for period in melody_grouped_by_period:
        if len(period[1]) > number_of_melody_notes_to_consider:
            important_melody_by_period.append([period[0], period[1][:number_of_melody_notes_to_consider]])
            
        else:
            important_melody_by_period.append(period)
            
    melody_by_period = pd.DataFrame(important_melody_by_period)
   
    del bar_number, bar_numbers, line_number, melody_grouped_by_period, melody_note, melody_notes_in_bar, note, melody, important_melody_by_period, period, number_of_melody_notes_to_consider
    
    return melody_by_period
    

def DissonanceCurve(melody_by_period):
    import pandas as pd
    import numpy as np
    
    last_period_number = int(melody_by_period.values[-1][0])
    
    periods = pd.concat([pd.Series(np.arange(1, last_period_number + 1)), pd.Series([11 for number in range(last_period_number)])], axis = 1)

    #For this version, I thought of the graph in a different manner
    #1. A - It still starts at 0 and ends at 0 but                                 [first and last periods]
    #4. B - the start has near-0 values,                                           [first 15%]
    #3. C - then it slowly goes up to 2/3-ish values with occasional high points,  [following 60%]
    #2. D - then it builds up to a dissonant climactic point with higher values,   [following 15%]
    #5. E - quickly falls to 1/2-ish values                                        [following 5%]
    #6. F - and reaches and stays at 0                                             [last 5%]

    #1. A
    periods.iloc[-1, 1] = 0
    
    if periods.shape[0] > 1:
        periods.iloc[0, 1] = 0

    #6. F
    for period in range(int((periods.shape[0] - 2) * 0.97), (int((periods.shape[0] - 2) * 0.97)) + (int((periods.shape[0] - 2) * 0.05)) + 1):
        periods.iloc[period, 1] = 0

    #5. E
    for period in range(int((periods.shape[0] - 2) * 0.95), (int((periods.shape[0] - 2) * 0.95)) + (int((periods.shape[0] - 2) * 0.05)) + 1):
        periods.iloc[period, 1] = np.random.randint(1, 2)
    
    #4. B
    for period in range(1, int((periods.shape[0] - 2) * 0.15) + 1):
        periods.iloc[period, 1] = np.random.randint(0, 1)

    #3. C
    for period in range(int((periods.shape[0] - 2) * 0.15), (int((periods.shape[0] - 2) * 0.15)) + (int((periods.shape[0] - 2) * 0.60)) + 1):
        periods.iloc[period, 1] = np.random.randint(1, 3) + round((2 * period) / (periods.shape[0]))

    #2. D
    for period in range(int((periods.shape[0] - 2) * 0.75), (int((periods.shape[0] - 2) * 0.75)) + (int((periods.shape[0] - 2) * 0.10)) + 1):
        periods.iloc[period, 1] = np.random.randint(2, 5)
        
    for period in range(int((periods.shape[0] - 2) * 0.85), (int((periods.shape[0] - 2) * 0.85)) + (int((periods.shape[0] - 2) * 0.10)) + 1):
        periods.iloc[period, 1] = np.random.randint(2, 4) + round((2 * period) / (periods.shape[0]))

    #Leftovers
    for period, leftovers in enumerate(periods.values):
        if leftovers[1] == 11:
            periods.iloc[period, 1] = np.random.randint(0, 1)

    period_tension_notes = pd.merge(periods, melody_by_period, how = "left", on = 0)
    
    del last_period_number, melody_by_period, period, periods, leftovers
    
    return period_tension_notes
    

def DissonanceMetric(list_of_notes):
    list_of_notes = sorted(list_of_notes)
    dissonance_list = [0, 18, 9, 0.5, 0.5, 0, 18, 0, 8.5, 3.5, 6.5, 15]
    
    interval_values = []
    
    for note_1_index, note_1 in enumerate(list_of_notes):
        for note_2_index, note_2 in enumerate(list_of_notes):
            if note_1_index < note_2_index:
                interval_values.append((note_2 - note_1) % 12)
    
    dissonance_values = []
    
    for interval in interval_values:
        dissonance_values.append(dissonance_list[interval])
    
    dissonance_metric = sum(dissonance_values) / len(list_of_notes)
    
    del dissonance_list, interval_values, note_1, note_1_index, note_2, note_2_index, list_of_notes, dissonance_values, interval
    
    return dissonance_metric


def CreateHarmonies(period, initial_harmonies, tolerance):
    import pandas as pd
    import random
    
    melody_notes, target = period[2], period[1]
    
    random_number = random.randint(1, 10)
    
    if random_number >= 1 and random_number <= 4: 
        harmony_ranges = [[initial_harmony, (initial_harmony - 1), (initial_harmony + 1), (initial_harmony - 2), (initial_harmony + 2)] for initial_harmony in initial_harmonies]

    elif random_number == 5 or random_number == 6: 
        harmony_ranges = [[(initial_harmony - 1), (initial_harmony - 2), initial_harmony, (initial_harmony + 1), (initial_harmony + 2)] for initial_harmony in initial_harmonies]

    elif random_number == 7 or random_number == 8: 
        harmony_ranges = [[(initial_harmony + 1), (initial_harmony + 2), initial_harmony, (initial_harmony - 1), (initial_harmony - 2)] for initial_harmony in initial_harmonies]

    elif random_number == 9 or random_number == 10: 
        harmony_ranges = [[(initial_harmony - 2), (initial_harmony + 2), (initial_harmony + 1), (initial_harmony - 1), initial_harmony] for initial_harmony in initial_harmonies]


    global last_melody_note
    
    if len(melody_notes) > 0:
        last_melody_note = melody_notes[0]
    
    
    def OneHarmony(melody_notes, target, harmony_ranges, tolerance):
        harmony_table = pd.DataFrame([["Harmony #1", "Dissonance Metric"]])
                                                   
        for harmony1 in harmony_ranges[0]:
            if harmony1 in melody_notes or harmony1 > last_melody_note:
                pass
            else:
                if len(melody_notes) == 0:
                    dissonance_metric = 0
                else:
                    dissonance_metric = sum([DissonanceMetric([harmony1] + [melody_notes[n]]) for n in range(len(melody_notes))]) / len(melody_notes)
                
                if abs(dissonance_metric - target) <= tolerance:
                    return [harmony1]
                
                else:                        
                    harmony_table = pd.concat((harmony_table, pd.DataFrame([[harmony1, dissonance_metric]])), axis = 0, ignore_index = True)
        return harmony_table
    
    
    def TwoHarmonies(melody_notes, target, harmony_ranges, tolerance):        
        harmony_table = pd.DataFrame([["Harmony #1", "Harmony #2", "Dissonance Metric"]])
                                       
        for harmony1 in harmony_ranges[0]:
            for harmony2 in harmony_ranges[1]:
                if harmony1 == harmony2 or harmony1 in melody_notes or harmony2 in melody_notes or harmony1 > last_melody_note or harmony2 > last_melody_note:
                    pass
                else:
                    if len(melody_notes) == 0:
                        dissonance_metric = DissonanceMetric([harmony1, harmony2])
                    else:
                        dissonance_metric = sum([DissonanceMetric([harmony1, harmony2] + [melody_notes[n]]) for n in range(len(melody_notes))]) / len(melody_notes)

                    if abs(dissonance_metric - target) <= tolerance:
                        return [harmony1, harmony2]
                    
                    else:                        
                        harmony_table = pd.concat((harmony_table, pd.DataFrame([[harmony1, harmony2, dissonance_metric]])), axis = 0, ignore_index = True)
        return harmony_table
    
    
    def ThreeHarmonies(melody_notes, target, harmony_ranges, tolerance):        
        harmony_table = pd.DataFrame([["Harmony #1", "Harmony #2", "Harmony #3", "Dissonance Metric"]])
                                       
        for harmony1 in harmony_ranges[0]:
            for harmony2 in harmony_ranges[1]:
                for harmony3 in harmony_ranges[2]:
                    if harmony1 == harmony2 or harmony1 == harmony3 or harmony1 in melody_notes or harmony2 in melody_notes or harmony2 == harmony3 or harmony3 in melody_notes or harmony1 > last_melody_note or harmony2 > last_melody_note or harmony3 > last_melody_note:
                        pass
                    else:
                        if len(melody_notes) == 0:
                            dissonance_metric = DissonanceMetric([harmony1, harmony2, harmony3])
                        else:
                            dissonance_metric = sum([DissonanceMetric([harmony1, harmony2, harmony3] + [melody_notes[n]]) for n in range(len(melody_notes))]) / len(melody_notes)
                        
                        if abs(dissonance_metric - target) <= tolerance:
                            return [harmony1, harmony2, harmony3]
                        
                        else:                        
                            harmony_table = pd.concat((harmony_table, pd.DataFrame([[harmony1, harmony2, harmony3, dissonance_metric]])), axis = 0, ignore_index = True)
        return harmony_table
    
    
    def FourHarmonies(melody_notes, target, harmony_ranges, tolerance):        
        harmony_table = pd.DataFrame([["Harmony #1", "Harmony #2", "Harmony #3", "Harmony #4", "Dissonance Metric"]])
                                       
        for harmony1 in harmony_ranges[0]:
            for harmony2 in harmony_ranges[1]:
                for harmony3 in harmony_ranges[2]:
                    for harmony4 in harmony_ranges[3]:
                        if harmony1 == harmony2 or harmony1 == harmony3 or harmony1 == harmony4 or harmony2 == harmony3 or harmony2 == harmony4 or harmony3 == harmony4 or harmony1 in melody_notes or harmony2 in melody_notes or harmony3 in melody_notes or harmony4 in melody_notes  or harmony1 > last_melody_note or harmony2 > last_melody_note or harmony3 > last_melody_note or harmony4 > last_melody_note:
                            pass
                        else:
                            if len(melody_notes) == 0:
                                dissonance_metric = DissonanceMetric([harmony1, harmony2, harmony3, harmony4])
                            else:
                                dissonance_metric = sum([DissonanceMetric([harmony1, harmony2, harmony3, harmony4] + [melody_notes[n]]) for n in range(len(melody_notes))]) / len(melody_notes)
                            
                            if abs(dissonance_metric - target) <= tolerance:
                                return [harmony1, harmony2, harmony3, harmony4]
                            
                            else:                        
                                harmony_table = pd.concat((harmony_table, pd.DataFrame([[harmony1, harmony2, harmony3, harmony4, dissonance_metric]])), axis = 0, ignore_index = True)
        return harmony_table
    
    
    def FiveHarmonies(melody_notes, target, harmony_ranges, tolerance):        
        harmony_table = pd.DataFrame([["Harmony #1", "Harmony #2", "Harmony #3", "Harmony #4", "Harmony #5", "Dissonance Metric"]])

        for harmony1 in harmony_ranges[0]:
            for harmony2 in harmony_ranges[1]:
                for harmony3 in harmony_ranges[2]:
                    for harmony4 in harmony_ranges[3]:
                        for harmony5 in harmony_ranges[4]:
                            if harmony1 == harmony2 or harmony1 == harmony3 or harmony1 == harmony4 or harmony2 == harmony3 or harmony2 == harmony4 or harmony3 == harmony4 or harmony1 in melody_notes or harmony2 in melody_notes or harmony3 in melody_notes or harmony4 in melody_notes or harmony1 == harmony5 or harmony2 == harmony5 or harmony3 == harmony5 or harmony4 == harmony5 or harmony5 in melody_notes or harmony1 > last_melody_note or harmony2 > last_melody_note or harmony3 > last_melody_note or harmony4 > last_melody_note or harmony5 > last_melody_note:
                                pass
                            else:
                                if len(melody_notes) == 0:
                                    dissonance_metric = DissonanceMetric([harmony1, harmony2, harmony3, harmony4, harmony5])
                                else:
                                    dissonance_metric = sum([DissonanceMetric([harmony1, harmony2, harmony3, harmony4, harmony5] + [melody_notes[n]]) for n in range(len(melody_notes))]) / len(melody_notes)
                                
                                if abs(dissonance_metric - target) <= tolerance:
                                    return [harmony1, harmony2, harmony3, harmony4, harmony5]
                                
                                else:                        
                                    harmony_table = pd.concat((harmony_table, pd.DataFrame([[harmony1, harmony2, harmony3, harmony4, harmony5, dissonance_metric]])), axis = 0, ignore_index = True)
        return harmony_table
    
    if len(initial_harmonies) == 1:
        return OneHarmony(melody_notes, target, harmony_ranges, tolerance)
    
    elif len(initial_harmonies) == 2:
        return TwoHarmonies(melody_notes, target, harmony_ranges, tolerance)
        
    elif len(initial_harmonies) == 3:
        return ThreeHarmonies(melody_notes, target, harmony_ranges, tolerance)
        
    elif len(initial_harmonies) == 4:
        return FourHarmonies(melody_notes, target, harmony_ranges, tolerance)
        
    elif len(initial_harmonies) == 5:
        return FiveHarmonies(melody_notes, target, harmony_ranges, tolerance)



def Harmonize(period_tension_notes, number_of_harmonies, tolerance, melody_minimum):
    import pandas as pd
    import numpy as np

    column_three = []

    for period in period_tension_notes.values:
        if str(period[2]) == str(np.nan):
            column_three.append([])
            
        else:
            column_three.append(period[2])
    
    period_tension_notes = pd.concat((period_tension_notes.iloc[:, :2], pd.Series(column_three)), axis = 1)
    
    initial_harmonies = [melody_minimum - (np.random.randint(low = 4, high = 10) * (iteration + 1)) for iteration in range(number_of_harmonies)]

    list_of_harmonies = []

    for period in period_tension_notes.values:
        harmonies = CreateHarmonies(period, initial_harmonies, tolerance)
        if isinstance(harmonies, pd.DataFrame):
            if len(harmonies.values) == 1:
                harmonies = initial_harmonies
            else:
                metrics = harmonies.iloc[1:]
                metrics = metrics.iloc[:, -1]
                metrics = abs(metrics - period[1]).astype('float64')
                minimum_metric_index = metrics.idxmin()
                
                harmonies = list(harmonies.values[minimum_metric_index])[:-1]
        
        initial_harmonies = harmonies
        
        list_of_harmonies.append([initial_harmonies])

    harmonized_melody = pd.concat((period_tension_notes, pd.DataFrame(list_of_harmonies)), axis = 1)

    del period_tension_notes, number_of_harmonies, column_three, harmonies, initial_harmonies, list_of_harmonies, period, tolerance
    
    return harmonized_melody


def HarmonizedFormToCsv(harmonized_melody, melody_original, harmonies_per_bar, harmony_velocities, tempo_value, time_signature_value):
    import pandas as pd
    import numpy as np
    
    formatted_harmonies = harmonized_melody.iloc[:, [0, 3]]
    
    formatted_harmonies_list = []
    
    for row_number, row in enumerate(formatted_harmonies.values):
        for note_number, note in enumerate(row[1]):
            formatted_harmonies_list.append([int((row_number * int(((time_signature_value[0]) / (2 ** time_signature_value[1])) * 1920) // harmonies_per_bar) + note_number + 1), note, harmony_velocities])
            formatted_harmonies_list.append([int(((row_number + 1) * int(((time_signature_value[0]) / (2 ** time_signature_value[1])) * 1920) // harmonies_per_bar) * 0.995) + note_number + 1, note, 0])
            
    formatted_harmonized_melody = pd.concat((pd.DataFrame(formatted_harmonies_list), melody_original), ignore_index = True)
    
    formatted_harmonized_melody = formatted_harmonized_melody.sort_values(by = [0]).reset_index(drop = True)
    
    
    formatted_dataframe = pd.concat((pd.Series(np.ones(formatted_harmonized_melody.shape[0], dtype = np.int8)), formatted_harmonized_melody.iloc[:, 0], pd.Series(["Note_on_c" for note in range(formatted_harmonized_melody.shape[0])]), pd.Series(np.zeros(formatted_harmonized_melody.shape[0], dtype = np.int8)), formatted_harmonized_melody.iloc[:, 1], formatted_harmonized_melody.iloc[:, 2], pd.Series(([""] * formatted_harmonized_melody.shape[0]))), axis = 1, ignore_index = True)
    
    initial_rows = pd.DataFrame([[0, 0, "Header", 1, 2, 480, ""], [1, 0, "Start_track", "", "", "", ""], [1, 0, "Time_signature", time_signature_value[0], time_signature_value[1], 24, 8], [1, 0, "Key_signature", 0, "major", "", ""], [1, 0, "Tempo", tempo_value, "", "", ""], [1, 0, "Control_c", 0, 121, 0, ""], [1, 0, "Program_c", 0, 0, "", ""], [1, 0, "Control_c", 0, 7, 100, ""], [1, 0, "Control_c", 0, 10, 64, ""], [1, 0, "Control_c", 0, 91, 0, ""], [1, 0, "Control_c", 0, 93, 0, ""], [1, 0, "MIDI_port", 0, "", "", ""]])
    
    last_rows = pd.DataFrame([[1, (formatted_dataframe.values[-1][1] + 1), "End_track", "", "", "", ""], [0, 0, "End_of_file", "", "", "", ""]])
        
    final_dataframe = pd.concat((initial_rows, formatted_dataframe, last_rows))

    final_dataframe = list(", ".join((str(row)[1:-1]).replace("'", "").replace(".0", "").split()) for row in final_dataframe.values)
    
    del formatted_harmonies, harmonized_melody, melody_original, harmonies_per_bar, formatted_harmonies_list, row_number, row, note, formatted_harmonized_melody, formatted_dataframe, initial_rows, last_rows, note_number, harmony_velocities, time_signature_value
    
    return final_dataframe


def CsvToMidi(final_dataframe, file):    
    import py_midicsv as pm
    
    midi_object = pm.csv_to_midi(final_dataframe)
    
    save_file_name = os.path.join("HarmonizedMelodies/", ("harmonized_" + file[18:]))
    
    with open(save_file_name, "wb") as output_file:
        midi_writer = pm.FileWriter(output_file)
        midi_writer.write(midi_object)
    
    del midi_object, final_dataframe
    
    return save_file_name


def Harmonizer(file, harmonies_per_bar, number_of_harmonies, tolerance, harmony_velocities, melody_velocities, number_of_melody_notes_to_consider):
    dataframe = MidiToCsv(file)
        
    melody, melody_original, melody_minimum, tempo_value, time_signature_value = ReFormatDataFrame(dataframe, melody_velocities)
    
    melody = LockingToGrid1(melody, harmonies_per_bar, time_signature_value)
    
    melody_by_period = LockingToGrid2(melody, number_of_melody_notes_to_consider)

    period_tension_notes = DissonanceCurve(melody_by_period)
    
    harmonized_melody = Harmonize(period_tension_notes, number_of_harmonies, tolerance, melody_minimum)

    final_dataframe = HarmonizedFormToCsv(harmonized_melody, melody_original, harmonies_per_bar, harmony_velocities, tempo_value, time_signature_value)

    return_file_name = CsvToMidi(final_dataframe, file)
    
    return return_file_name



from flask import Flask, render_template, request, send_file, redirect
from time import strftime
import os
import sys
import logging


if not os.path.exists("HarmonizedMelodies"):
    os.mkdir("HarmonizedMelodies")

if not os.path.exists("SubmittedMelodies"):
    os.mkdir("SubmittedMelodies")

app = Flask(__name__, static_url_path="/static")

@app.route("/")
def home():
    return render_template("about.html")

@app.route("/main")
def main():
    global filename
    filename = "NOPENOTGONNAWORKNOWAMI"
    
    return render_template("home.html", message="")

@app.route('/main-list', methods = ['GET', 'POST'])
def list_file():
    global original_filename, filename

    if request.method == 'POST':
        
        original_filename = request.form["choice_from_list"]
        
        filename = "melody_from_list"
        
        return render_template("home.html", message=original_filename)


@app.route('/main-upload', methods = ['GET', 'POST'])
def upload_file():
    global original_filename    

    if request.method == 'POST':
        try:
            f = request.files['file']

            global filename
          
            filename = strftime("%a_%d_%b_%Y_%H_%M_%S.mid")
            
            f.save(os.path.join("SubmittedMelodies/", filename))
            
            if f.filename[-4:] == ".mid":
                original_filename = f.filename[:-4]
            elif f.filename[-4:] == "midi":
                original_filename = f.filename[:-5]

            return render_template("home.html", message=original_filename)
        
        except KeyError:
            return render_template("home.html", message="Please upload a MIDI file!")
        
    
@app.route('/harmonize', methods = ['GET', 'POST'])
def harmonize():
    if request.method == 'POST':
        if filename == "NOPENOTGONNAWORKNOWAMI":
            return render_template("home.html", message="Please upload a MIDI file!")
        
        if filename == "melody_from_list":
            file = os.path.join("StandardMelodies/", original_filename)
            file = file + ".mid"
        
        else:
            file = os.path.join("SubmittedMelodies/", filename)
        
        global return_file_name
        return_file_name = Harmonizer(file = file, harmonies_per_bar = float(request.form["number_of_harmonies_per_bar"]), number_of_harmonies = int(request.form["number_of_harmony_notes"]), tolerance = float(request.form["tolerance"]), harmony_velocities = int(request.form["harmony velocity"]), melody_velocities = int(request.form["melody velocity"]), number_of_melody_notes_to_consider = int(request.form["relevant notes"]))
        
        return render_template("harmonized.html")
    
    
@app.route('/Harmonized-Melody-Redirect/')
def HarmonizedMelodyRedirect():
    return redirect("/Harmonized-Melody")

@app.route('/Harmonized-Melody')
def HarmonizedMelody():
    return send_file(filename_or_fp = return_file_name, as_attachment=True, attachment_filename=(original_filename + " - Harmonized.mid"), cache_timeout = 0)
    
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)
    
    