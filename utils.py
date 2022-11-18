from constant import RESULT_COLOR_HIST_FILE_PATH, RESULT_CATEGORIES


def save_result(file, time_create_hist_ref, time, distances, top, winner, test_ok):
    f = open(RESULT_COLOR_HIST_FILE_PATH, "a")
    f.write( "Requete ;" + file + "\n\n")

    f.write( "Temps de création des histogrammes de références ;" + str(time_create_hist_ref) + "\n")
    f.write( "Temps pour appliquer l'algorithme de reconnaissance d'image ;" + str(time) + "\n\n")

    f.write("Résultat de la reconnaissance :\n")
    f.write("Position;Fichier;Distance\n")
    pos = 1
    for key, value in distances.items():
        f.write(str(pos) + ";" + key + ";" + str(value) + "\n")
        pos = pos + 1

    f.write("\nTop " + str(len(top)) + " :\n")
    f.write("Position;Fichier;Distance\n")
    pos = 1
    for key, value in top.items():
        f.write(str(pos) + ";" + key + ";" + str(value) + "\n")
        pos = pos + 1

    if winner is None:
        winner = "Inconnu"
    f.write("\nRésultat obtenu ;" + winner)
    resul_wanted = RESULT_CATEGORIES[file]
    if RESULT_CATEGORIES[file] is None:
        resul_wanted = "Inconnu"
    f.write("\nRésultat attendu ;" + resul_wanted )
    f.write("\nTest OK ;" + str(test_ok) + "\n\n")

    f.close()