import sqlite3
import numpy as np
from tokenize import Name
import io

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

#Erzeugt eine Klasse definiert folgend:
class Datenbanken:
    #Starte die Klasse mit dem namen der Datenbank
    def __init__(self,name) -> None:
        self.__connection = sqlite3.connect(name, detect_types=sqlite3.PARSE_DECLTYPES)
        self.__cursor = self.__connection.cursor()
    #Wechsle zu einer anderen Datenbank (Anderer File) (Datenbank muss existieren sonst fehler meldung)
    def Datenbankwechsel(self,name):
        nameaslist = [name]
        res = self.__cursor.execute("SELECT name FROM sqlite_master WHERE name=?", nameaslist)
        if res.fetchone() is None:
            return("Diese Datenbank existiert noch nicht")
        else:
            self.__connection = sqlite3.connect(name, detect_types=sqlite3.PARSE_DECLTYPES)
            self.__cursor = self.__connection.cursor()
    #Erstelle eine Datenbank falls sie noch nicht existiert sonst fehlermeldung
    def DatenbankErstellen(self,name):
        nameaslist = [name]
        res = self.__cursor.execute("SELECT name FROM sqlite_master WHERE name=?", nameaslist)
        if res.fetchone() is None:
            self.__connection = sqlite3.connect(name, detect_types=sqlite3.PARSE_DECLTYPES)
            self.__cursor = self.__connection.cursor()
        else:
            return("Diese Datenbank existiert schon")
    #Gebe eine liste aller tabellen in einer Datenbank aus
    def ListeTabellen(self):
        return(self.__cursor.execute("SELECT name FROM sqlite_master").fetchall())
    #Suche eine Tabelle mit angegebenen namen in der derzeitigen Datenbank
    def SucheTabelle(self,name):
        nameaslist = [name]
        res = self.__cursor.execute("SELECT name FROM sqlite_master WHERE name=?", nameaslist)
        return(res.fetchone())
    #Erstelle eine Tabelle in der derzeitigen Datenbank
    #Collums ist eine liste mit allen Collums der Datenbank die bestehen aus collum name und leerer string / "array" um zu zeigen ob normale objekte oder arrays
    #in diesem Collum sind. ein Array collum speichert nur arrays und ein nicht array collum kann keine arrays speichern.
    def TabellenErstellen(self,name,collums):
        if self.SucheTabelle(name) == None:
            String = "CREATE TABLE "+name
            temp = ""
            for i in collums:
                temp = temp+i[0]+" "+i[1]+","
            temp = temp[:-1]
            self.__cursor.execute(String+"("+temp+")")
    #Füge das Datenset in die angegebene Tabelle ein, data ist eine Liste aller Daten, die in der reinfolge so in the collums eingetragen werden
    def TabellenInsert(self,name,data):
        length = len(data)
        string = "INSERT INTO "+name+" VALUES"
        print(data)
        self.__cursor.execute(string+"("+(length-1)*"?, "+" ?"+")",data)
        self.__connection.commit()
    #Lösche aus einer angebebene Tabelle alle zeilen mit den angegeben bedingungen. (Bedingungscollums, -zeichen, bedingungen sind alles listen mit dem i'ten
    #element der bedingungscollum liste dem i'ten element aus der bedingungszeichen liste und dem i'ten element aus der bedingungs liste als statement der i'ten
    #bedingung (zb ["number1","number2"],[">","<="],[3,7] ==> number1 > 3 AND number2 <= 7) 
    def TabellenDelete(self,name,bedinungscollums,bedingungszeichen,bedingungen):
        string = "DELETE FROM "+name+" WHERE "
        for i in range(0,len(bedinungscollums)):
            string = string + bedinungscollums[i]+" "+bedingungszeichen[i]+" ? AND "
        string = string[:-5]
        self.__cursor.execute(string,bedingungen)
    #Lösche eine gesamte tabelle (wahrscheinlich nicht benötigt)
    def TabellusDeletius(self,name):
        self.__cursor.execute("DROP TABLE IF EXISTS "+name)
    #In einer tabelle mit gleichen bedinungen wie bei TabellenDelete, setze die collums angegeben in "whatnew" zu den werten angegeben in "new"
    def TabelleUpdaten(self,name,whatnew,new,bedinungscollums,bedingungszeichen,bedingungen):
        string = "UPDATE "+name+" SET "
        for j in range (0,len(whatnew)):
            string = string + whatnew[j] +" = ? , "
        string = string[:-2]
        string = string+"WHERE "
        for i in range(0,len(bedinungscollums)):
            string = string + bedinungscollums[i]+" "+bedingungszeichen[i]+" ? AND "
        string = string[:-5]
        new.extend(bedingungen)
        self.__cursor.execute(string,new)
        self.__connection.commit()
    #gebe eine gesamte tabelle aus
    def Tabelleabrufen(self,name,sortierung):
        return(self.__cursor.execute("SELECT * FROM "+name+" ORDER BY "+sortierung).fetchall())
    #Gebe die angegeben collums aller zeilen aus die die bedingungen erfüllen
    def VonTabelleGebe(self,name,collums,sortierung,bedinungscollums,bedingungszeichen,bedingungen):
        string = "SELECT "
        for k in collums:
            string = string +k +" , "
        string = string[:-2]
        string = string+"FROM "+name+" WHERE "
        for i in range(0,len(bedinungscollums)):
            string = string + bedinungscollums[i]+" "+bedingungszeichen[i]+" ? AND "
        string = string[:-5]
        string = string+" ORDER BY "+sortierung
        return(self.__cursor.execute(string,bedingungen).fetchall())

def unit_test():
    d = Datenbanken("hello")
    print(d.Datenbankwechsel("nuh uh"))
    d.DatenbankErstellen("hello")
    d.DatenbankErstellen("nuh uh")
    print(d.Datenbankwechsel("nuh uh"))
    print(d.ListeTabellen())
    d.TabellenErstellen("bitches", [["hello",""],["nuhuh",""],["fuckyou","array"]])
    print(d.ListeTabellen())
    d.TabellenInsert("bitches",["suck","ma","aick"])
    d.TabelleUpdaten("bitches",["fuckyou","nuhuh"],[np.random.rand(10, 10),"Ich Funktioniere"],["fuckyou","hello"],[">=","="],["7","suck"])
    print(d.ListeTabellen())
    print(d.Tabelleabrufen("bitches","hello")) 
    print(d.VonTabelleGebe("bitches",["hello","nuhuh"],"nuhuh",["nuhuh"],["=="],["Ich Funktioniere"]))
    d.TabellusDeletius("bitches")
if __name__ == "__main__":
    unit_test()
