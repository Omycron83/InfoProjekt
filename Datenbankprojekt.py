import sqlite3
import numpy as np
from tokenize import Name

class Datenbanken:
    def __init__(self,name) -> None:
        self.__connection = sqlite3.connect(name)
        self.__cursor = self.__connection.cursor()
    def Datenbankwechsel(self,name):
        nameaslist = [name]
        res = self.__cursor.execute("SELECT name FROM sqlite_master WHERE name=?", nameaslist)
        if res.fetchone() is None:
            return("Diese Datenbank existiert noch nicht")
        else:
            self.__connection = sqlite3.connect(name)
            self.__cursor = self.__connection.cursor()
    def DatenbankErstellen(self,name):
        nameaslist = [name]
        res = self.__cursor.execute("SELECT name FROM sqlite_master WHERE name=?", nameaslist)
        if res.fetchone() is None:
            self.__connection = sqlite3.connect(name)
            self.__cursor = self.__connection.cursor()
        else:
            return("Diese Datenbank existiert schon")
    def ListeTabellen(self):
        return(self.__cursor.execute("SELECT name FROM sqlite_master").fetchall())
    def SucheTabelle(self,name):
        nameaslist = [name]
        res = self.__cursor.execute("SELECT name FROM sqlite_master WHERE name=?", nameaslist)
        return(res.fetchone())
    def TabellenErstellen(self,name,collums):
        if self.SucheTabelle(name) == None:
            String = "CREATE TABLE "+name
            temp = ""
            for i in collums:
                temp = temp+i+" ,"
            temp = temp[:-1]
            self.__cursor.execute(String+"("+temp+")")
    def TabellenInsert(self,name,data):
        length = len(data)
        string = "INSERT INTO "+name+" VALUES"
        print(data)
        self.__cursor.execute(string+"("+(length-1)*"?, "+" ?"+")",data)
        self.__connection.commit()
    def TabellenDelete(self,name,bedinungscollums,bedingungszeichen,bedingungen):
        string = "DELETE FROM "+name+" WHERE "
        for i in range(0,len(bedinungscollums)):
            string = string + bedinungscollums[i]+" "+bedingungszeichen[i]+" ? AND "
        string = string[:-5]
        self.__cursor.execute(string,bedingungen)
    def TabellenMassdelete(self,data):
        for i in data:
            self.TabellenDelete(i[0],i[1],i[2],i[3])
    def TabellusDeletius(self,name):
        self.__cursor.execute("DROP TABLE IF EXISTS "+name)
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
    def Tabelleabrufen(self,name,sortierung):
        return(self.__cursor.execute("SELECT * FROM "+name+" ORDER BY "+sortierung).fetchall())


d = Datenbanken("hello")
print(d.Datenbankwechsel("nuh uh"))
d.DatenbankErstellen("hello")
d.DatenbankErstellen("nuh uh")
print(d.Datenbankwechsel("nuh uh"))
print(d.ListeTabellen())
d.TabellenErstellen("bitches", ["hello","nuhuh","fuckyou"])
print(d.ListeTabellen())
d.TabellenInsert("bitches",["suck","ma","aick"])
d.TabelleUpdaten("bitches",["fuckyou","nuhuh"],[np.array([5]),"Ich Funktioniere"],["fuckyou","hello"],[">=","="],["7","suck"])
print(d.ListeTabellen())
print(d.Tabelleabrufen("bitches","fuckyou")) 
d.TabellusDeletius("bitches")