import numpy as np
import random as r
from plots import plot_scatter
from plots import plot_function_values


class ApstraktnaIndividua:
    def __init__(self, DuzinaHromozoma, kriterij, VelicinaProblema, opseg):
        self.DuzinaHromozoma = DuzinaHromozoma
        self.Hromozom = np.random.randint(2, size=DuzinaHromozoma)
        self.kriterij = kriterij
        self.VelicinaProblema = VelicinaProblema
        self.opseg = opseg
        self.Evaluiraj()

    def SetDuzinaHromozoma(self, DuzinaHromozoma):
        self.DuzinaHromozoma = DuzinaHromozoma

    def GetDuzinaHromozoma(self):
        return self.DuzinaHromozoma

    def SetHromozom(self, Hromozom):
        self.Hromozom = Hromozom

    def GetHromozom(self):
        return self.Hromozom

    def SetFitness(self, Fitness):
        self.Fitness = Fitness

    def GetFitness(self):
        return self.Fitness

    def Evaluiraj(self): pass


def OdrediP(opseg):
    max_granica = max(abs(opseg[0]), abs(opseg[1]))
    return int(np.log2(max_granica))


def KodirajHromozom(r1, a, opseg):
    p = OdrediP(opseg)
    x = []
    i = 0
    while i < len(a):
        predznak = a[i]
        cijeli_dio = BinToDec(a[i + 1:i + 1 + p])
        decimalni_dio = BinToDec1(a[i + 1 + p:i + r1])
        res = cijeli_dio + decimalni_dio
        if predznak == 1:
            res *= -1
        x.append(res)
        i += r1
    return x


class MojaIndividua(ApstraktnaIndividua):
    def Evaluiraj(self):
        if self.DuzinaHromozoma % self.VelicinaProblema != 0 or self.DuzinaHromozoma < 4:
            raise Exception("Neispravna duzina hromozoma!")
        d = self.DuzinaHromozoma
        r1 = int(d / self.VelicinaProblema)
        a = self.Hromozom
        x = KodirajHromozom(r1, a, self.opseg)
        self.SetFitness(self.kriterij(x))


class Populacija:
    def __init__(self, kriterij, opseg, VelicinaProblema, RangSP, VelicinaPopulacije, VjerovatnocaUkrstanja,
                 VjerovatnocaMutacije, MaxGeneracije, VelicinaElite,
                 DuzinaHromozoma, plot):
        self.kriterij = kriterij
        self.opseg = opseg
        self.VelicinaProblema = VelicinaProblema
        self.RangSP = RangSP

        self.VelicinaPopulacije = VelicinaPopulacije
        self.SetVjerovatnocaUkrstanja(VjerovatnocaUkrstanja)
        self.SetVjerovatnocaMutacije(VjerovatnocaMutacije)
        self.MaxGeneracija = MaxGeneracije
        self.SetVelicinaElite(VelicinaElite)
        self.DuzinaHromozoma = DuzinaHromozoma
        self.Populacija = []
        for i in range(0, VelicinaPopulacije):
            individua = MojaIndividua(DuzinaHromozoma, kriterij, VelicinaProblema, opseg)
            self.Populacija.append(individua)
        self.plot = plot

    def GetPopulacija(self):
        return self.Populacija

    def GetFitnessPopulacije(self):
        return max(self.Populacija, key=lambda hromozom: hromozom.GetFitness()).GetFitness()

    def SetVelicinaPopulacije(self, VelicinaPopulacije):
        self.VelicinaPopulacije = VelicinaPopulacije

    def GetVelicinaPopulacije(self):
        return self.VelicinaPopulacije

    def SetVjerovatnocaUkrstanja(self, VjerovatnocaUkrstanja):
        if VjerovatnocaUkrstanja < 0 or VjerovatnocaUkrstanja > 1:
            raise Exception("Vrijednost vjerovatnoce mora biti izmedju 0 i 1!")
        self.VjerovatnocaUkrstanja = VjerovatnocaUkrstanja

    def GetVjerovatnocaUkrstanja(self):
        return self.VjerovatnocaUkrstanja

    def SetVjerovatnocaMutacije(self, VjerovatnocaMutacije):
        if VjerovatnocaMutacije < 0 or VjerovatnocaMutacije > 1:
            raise Exception("Vrijednost vjerovatnoce mora biti izmedju 0 i 1!")
        self.VjerovatnocaMutacije = VjerovatnocaMutacije

    def GetVjerovatnocaMutacije(self):
        return self.VjerovatnocaMutacije

    def SetMaxGeneracija(self, MaxGeneracija):
        self.MaxGeneracija = MaxGeneracija

    def GetMaxGeneracija(self):
        return self.MaxGeneracija

    def SetVelicinaElite(self, VelicinaElite):
        if VelicinaElite < 0 or VelicinaElite > self.VelicinaProblema:
            raise Exception("Velicina elite mora biti izmedju 0 i", self.VelicinaProblema)
        self.VelicinaElite = VelicinaElite

    def GetVelicinaElite(self):
        return self.VelicinaElite

    def OpUkrstanjeTacka(self, individua1, individua2):

        if self.VjerovatnocaUkrstanja > r.random():
            index_ukrstanja = r.randint(1, individua1.GetDuzinaHromozoma() - 1)
            hromozom1 = []
            hromozom2 = []
            hromozom1[:index_ukrstanja] = individua1.GetHromozom()[:index_ukrstanja]
            hromozom1[index_ukrstanja:] = individua2.GetHromozom()[
                                          index_ukrstanja:]  # drugi dio se uzme od drugog hromozoma

            hromozom2[:index_ukrstanja] = individua2.GetHromozom()[:index_ukrstanja]
            hromozom2[index_ukrstanja:] = individua1.GetHromozom()[index_ukrstanja:]

            c1 = MojaIndividua(individua1.GetDuzinaHromozoma(), self.kriterij, self.VelicinaProblema, self.opseg)
            c2 = MojaIndividua(individua2.GetDuzinaHromozoma(), self.kriterij, self.VelicinaProblema, self.opseg)
            c1.SetHromozom(hromozom1)
            c2.SetHromozom(hromozom2)
            c1.Evaluiraj()
            c2.Evaluiraj()
            return c1, c2

        return individua1, individua2

    def OpUkrstanjeDvijeTacke(self, individua1, individua2):

        if self.VjerovatnocaUkrstanja > r.random():
            index_ukrstanja1 = r.randint(1, individua1.GetDuzinaHromozoma() / 2 - 1)
            index_ukrstanja2 = r.randint(individua1.GetDuzinaHromozoma() / 2, individua1.GetDuzinaHromozoma() - 1)

            h1 = individua1.GetHromozom()
            h2 = individua2.GetHromozom()

            hromozom1 = []
            hromozom1[:index_ukrstanja1] = h1[:index_ukrstanja1]
            hromozom1[index_ukrstanja1:index_ukrstanja2] = h2[index_ukrstanja1:index_ukrstanja2]
            hromozom1[index_ukrstanja2:] = h1[index_ukrstanja2:]

            hromozom2 = []
            hromozom2[:index_ukrstanja1] = h2[:index_ukrstanja1]
            hromozom2[index_ukrstanja1:index_ukrstanja2] = h1[index_ukrstanja1:index_ukrstanja2]
            hromozom2[index_ukrstanja2:] = h2[index_ukrstanja2:]

            c1 = MojaIndividua(individua1.GetDuzinaHromozoma(), self.kriterij, self.VelicinaProblema, self.opseg)
            c2 = MojaIndividua(individua2.GetDuzinaHromozoma(), self.kriterij, self.VelicinaProblema, self.opseg)
            c1.SetHromozom(hromozom1)
            c2.SetHromozom(hromozom2)
            c1.Evaluiraj()
            c2.Evaluiraj()
            return c1, c2

        return individua1, individua2

    def OpBinMutacije(self, individua):
        hromozom = individua.GetHromozom()
        index = r.randint(1, len(hromozom) - 1)
        if self.VjerovatnocaMutacije > r.random():
            if hromozom[index] == 0:
                hromozom[index] = 1
            else:
                hromozom[index] = 0
            individua.SetHromozom(hromozom)
            individua.Evaluiraj()
        return individua

    def SelekcijaRTocak(self):
        j = 0
        u = r.random()
        suma_fitnesa = sum(hromozom.GetFitness() for hromozom in self.Populacija)
        suma = self.Populacija[j].GetFitness() / suma_fitnesa
        while suma < u:
            j += 1
            suma += self.Populacija[j].GetFitness() / suma_fitnesa
        return j

    def ModificiraniFitnes(self, hromozom):
        populacija_sorted = sorted(self.Populacija, key=lambda hromozom: hromozom.GetFitness(), reverse=True)
        return 2 - self.RangSP + 2 * (self.RangSP - 1) * (populacija_sorted.index(hromozom) - 1) / (
                self.VelicinaPopulacije - 1)

    def SelekcijaRang(self):
        j = 0
        u = r.random()
        suma_fitnesa = sum(self.ModificiraniFitnes(hromozom) for hromozom in self.Populacija)
        suma = self.ModificiraniFitnes(self.Populacija[j]) / suma_fitnesa
        while suma < u:
            j += 1
            suma += self.ModificiraniFitnes(self.Populacija[j]) / suma_fitnesa
        return j

    def NovaGeneracija(self):
        nova_populacija = []
        while len(nova_populacija) < self.VelicinaPopulacije:
            index1 = self.SelekcijaRang()
            index2 = self.SelekcijaRang()
            while index1 == index2:
                index2 = self.SelekcijaRang()
            h1 = self.Populacija[index1]
            h2 = self.Populacija[index2]
            (c1, c2) = self.OpUkrstanjeTacka(h1, h2)
            nova_populacija.append(c1)
            if len(nova_populacija) < self.VelicinaPopulacije:
                nova_populacija.append(c2)

        index_mutacije = r.randint(0, self.VelicinaPopulacije - 1)
        mutirana_individua = self.OpBinMutacije(nova_populacija[index_mutacije])
        nova_populacija[index_mutacije] = mutirana_individua

        nova_populacija_sorted = sorted(nova_populacija, key=lambda hromozom: hromozom.GetFitness(), reverse=False)
        populacija_sorted = sorted(self.Populacija, key=lambda hromozom: hromozom.GetFitness(), reverse=False)

        self.Populacija = populacija_sorted[:self.VelicinaElite]
        self.Populacija[self.VelicinaElite:] = nova_populacija_sorted[0:len(nova_populacija) - self.VelicinaElite]
        self.Populacija = sorted(self.Populacija, key=lambda hromozom: hromozom.GetFitness(), reverse=False)

    def GenerisiGeneracije(self):
        i = 0
        function_values = []
        while i < self.MaxGeneracija:
            self.NovaGeneracija()
            najbolji = self.Populacija[0]

            function_values.append(najbolji.GetFitness())

            if self.plot and (i == 0 or i == int(self.MaxGeneracija / 3) or i == int(
                    2 * self.MaxGeneracija / 3) or i == self.MaxGeneracija - 1):
                x_draw = [KodirajHromozom(int(individua.GetDuzinaHromozoma() / self.VelicinaProblema),
                                          individua.GetHromozom(), self.opseg)[0] for individua in self.Populacija]
                y_draw = [KodirajHromozom(int(individua.GetDuzinaHromozoma() / self.VelicinaProblema),
                                          individua.GetHromozom(), self.opseg)[1] for individua in self.Populacija]
                result = [x_draw, y_draw]
                plot_scatter(result, KodirajHromozom(int(najbolji.GetDuzinaHromozoma() / self.VelicinaProblema),
                                                     najbolji.GetHromozom(),
                                                     self.opseg), self.opseg)
                print("Najbolji fitnes u generaciji", i + 1, " je:", najbolji.GetFitness(), "u tacki:",
                KodirajHromozom(int(najbolji.GetDuzinaHromozoma() / self.VelicinaProblema), najbolji.GetHromozom(),
                self.opseg))
            i += 1
        self.Populacija = sorted(self.Populacija, key=lambda el: el.GetFitness(), reverse=False)

        if self.plot:
            plot_function_values(function_values, 'Geneticki algoritam')

        return KodirajHromozom(int(self.Populacija[0].GetDuzinaHromozoma() / self.VelicinaProblema),
                               self.Populacija[0].GetHromozom(), self.opseg), self.Populacija[0].GetFitness()


def BinToDec(a):
    res = 0
    for i in range(0, len(a)):
        res += a[i] * 2 ** (len(a) - 1 - i)
    return res


def BinToDec1(a):
    res = 0
    for i in range(0, len(a)):
        res += a[i] * 2 ** -(1 + i)
    return res
