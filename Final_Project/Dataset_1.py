import pandas as pd
from tqdm import tqdm

keys = [
    "teamname1", 
    "Aatrox1", "Ahri1", "Akali1", "Alistar1", "Amumu1", "Anivia1",  "Annie1", "Ashe1", "Aurelion Sol1", "Azir1",
    "Bard1", "Blitzcrank1", "Brand1", "Braum1",
    "Caitlyn1", "Camille1", "Cassiopeia1", "Cho'Gath1", "Corki1",
    "Darius1", "Diana1", "Dr. Mundo1", "Draven1",
    "Ekko1", "Elise1", "Evelynn1", "Ezreal1",
    "Fiddlesticks1", "Fiora1", "Fizz1", 
    "Galio1", "Gangplank1", "Garen1", "Gnar1", "Gragas1", "Graves1",
    "Hecarim1", "Heimerdinger1",
    "Illaoi1", "Irelia1", "Ivern1",
    "Janna1", "Jarvan IV1", "Jax1", "Jayce1", "Jhin1", "Jinx1",
    "Kai'Sa1", "Kalista1", "Karma1", "Karthus1", "Kassadin1", "Katarina1", "Kayle1", "Kayn1", "Kennen1", "Kha'Zix1", "Kindred1", "Kled1", "Kog'Maw1",
    "LeBlanc1", "Lee Sin1", "Leona1", "Lissandra1", "Lucian1", "Lulu1", "Lux1",
    "Malphite1", "Malzahar1", "Maokai1", "Master Yi1", "Miss Fortune1", "Mordekaiser1", "Morgana1",
    "Nami1", "Nasus1", "Nautilus1", "Neeko1", "Nidalee1", "Nocturne1", "Nunu & Willump1",
    "Olaf1", "Orianna1", "Ornn1",
    "Pantheon1", "Poppy1", "Pyke1",
    "Qiyana1", "Quinn1",
    "Rakan1", "Rammus1", "Rek'Sai1", "Renekton1", "Rengar1", "Riven1", "Rumble1", "Ryze1",
    "Sejuani1", "Shaco1", "Shen1", "Shyvana1", "Singed1", "Sion1", "Sivir1", "Skarner1", "Sona1", "Soraka1", "Swain1", "Sylas1", "Syndra1",
    "Tahm Kench1", "Taliyah1", "Talon1", "Taric1", "Teemo1", "Thresh1", "Tristana1", "Trundle1", "Tryndamere1", "Twisted Fate1", "Twitch1",
    "Udyr1", "Urgot1",
    "Varus1", "Vayne1", "Veigar1", "Vel'Koz1", "Vi1", "Viktor1", "Vladimir1", "Volibear1",
    "Warwick1", "Wukong1",
    "Xayah1", "Xerath1", "Xin Zhao1",
    "Yasuo1", "Yorick1", "Yuumi1",
    "Zac1", "Zed1", "Ziggs1", "Zilean1", "Zoe1", "Zyra1",
    "kills1",
    "assists1",
    "TKPM1",
    "CKPM1",
    "infernal1",
    "mountain1",
    "cloud1",
    "ocean1",
    "chemtech1",
    "hextech1",
    "elders1",
    "heralds1",
    "barons1",
    "towers1",
    "turrets1",
    "inhibitor1",
    "damage1",
    "DPM1",
    "DTPM1",
    "DMPM1",
    "wards1",
    "WPM1",
    "WK1",
    "WKPM1",
    "CW1",
    "VS1",
    "VSPM1",
    "Gold1",
    "EG1",
    "EGPM1",
    "EGS1",
    "GS1",
    "GSPD1",
    "MK1",
    "Ga101",
    "XPa101",
    "Ka101",
    "Aa101",
    "Ga151",
    "XPa151",
    "Ka151",
    "Aa151", 
    "teamname2",
    "Aatrox2", "Ahri2", "Akali2", "Alistar2", "Amumu2", "Anivia2",  "Annie2", "Ashe2", "Aurelion Sol2", "Azir2",
    "Bard2", "Blitzcrank2", "Brand2", "Braum2",
    "Caitlyn2", "Camille2", "Cassiopeia2", "Cho'Gath2", "Corki2",
    "Darius2", "Diana2", "Dr. Mundo2", "Draven2",
    "Ekko2", "Elise2", "Evelynn2", "Ezreal2",
    "Fiddlesticks2", "Fiora2", "Fizz2", 
    "Galio2", "Gangplank2", "Garen2", "Gnar2", "Gragas2", "Graves2",
    "Hecarim2", "Heimerdinger2",
    "Illaoi2", "Irelia2", "Ivern2",
    "Janna2", "Jarvan IV2", "Jax2", "Jayce2", "Jhin2", "Jinx2",
    "Kai'Sa2", "Kalista2", "Karma2", "Karthus2", "Kassadin2", "Katarina2", "Kayle2", "Kayn2", "Kennen2", "Kha'Zix2", "Kindred2", "Kled2", "Kog'Maw2",
    "LeBlanc2", "Lee Sin2", "Leona2", "Lissandra2", "Lucian2", "Lulu2", "Lux2",
    "Malphite2", "Malzahar2", "Maokai2", "Master Yi2", "Miss Fortune2", "Mordekaiser2", "Morgana2",
    "Nami2", "Nasus2", "Nautilus2", "Neeko2", "Nidalee2", "Nocturne2", "Nunu & Willump2",
    "Olaf2", "Orianna2", "Ornn2",
    "Pantheon2", "Poppy2", "Pyke2",
    "Qiyana2", "Quinn2",
    "Rakan2", "Rammus2", "Rek'Sai2", "Renekton2", "Rengar2", "Riven2", "Rumble2", "Ryze2",
    "Sejuani2", "Shaco2", "Shen2", "Shyvana2", "Singed2", "Sion2", "Sivir2", "Skarner2", "Sona2", "Soraka2", "Swain2", "Sylas2", "Syndra2",
    "Tahm Kench2", "Taliyah2", "Talon2", "Taric2", "Teemo2", "Thresh2", "Tristana2", "Trundle2", "Tryndamere2", "Twisted Fate2", "Twitch2",
    "Udyr2", "Urgot2",
    "Varus2", "Vayne2", "Veigar2", "Vel'Koz2", "Vi2", "Viktor2", "Vladimir2", "Volibear2",
    "Warwick2", "Wukong2",
    "Xayah2", "Xerath2", "Xin Zhao2",
    "Yasuo2", "Yorick2", "Yuumi2",
    "Zac2", "Zed2", "Ziggs2", "Zilean2", "Zoe2", "Zyra2",
    "kills2",
    "assists2",
    "TKPM2",
    "CKPM2",
    "infernal2",
    "mountain2",
    "cloud2",
    "ocean2",
    "chemtech2",
    "hextech2",
    "elders2",
    "heralds2",
    "barons2",
    "towers2",
    "turrets2",
    "inhibitor2",
    "damage2",
    "DPM2",
    "DTPM2",
    "DMPM2",
    "wards2",
    "WPM2",
    "WK2",
    "WKPM2",
    "CW2",
    "VS2",
    "VSPM2",
    "Gold2",
    "EG2",
    "EGPM2",
    "EGS2",
    "GS2",
    "GSPD2",
    "MK2",
    "Ga102",
    "XPa102",
    "Ka102",
    "Aa102",
    "Ga152",
    "XPa152",
    "Ka152",
    "Aa152",
    "Aatroxb", "Ahrib", "Akalib", "Alistarb", "Amumub", "Aniviab",  "Annieb", "Asheb", "Aurelion Solb", "Azirb",
    "Bardb", "Blitzcrankb", "Brandb", "Braumb",
    "Caitlynb", "Camilleb", "Cassiopeiab", "Cho'Gathb", "Corkib",
    "Dariusb", "Dianab", "Dr. Mundob", "Dravenb",
    "Ekkob", "Eliseb", "Evelynnb", "Ezrealb",
    "Fiddlesticksb", "Fiorab", "Fizzb", 
    "Galiob", "Gangplankb", "Garenb", "Gnarb", "Gragasb", "Gravesb",
    "Hecarimb", "Heimerdingerb",
    "Illaoib", "Ireliab", "Ivernb",
    "Jannab", "Jarvan IVb", "Jaxb", "Jayceb", "Jhinb", "Jinxb",
    "Kai'Sab", "Kalistab", "Karmab", "Karthusb", "Kassadinb", "Katarinab", "Kayleb", "Kaynb", "Kennenb", "Kha'Zixb", "Kindredb", "Kledb", "Kog'Mawb",
    "LeBlancb", "Lee Sinb", "Leonab", "Lissandrab", "Lucianb", "Lulub", "Luxb",
    "Malphiteb", "Malzaharb", "Maokaib", "Master Yib", "Miss Fortuneb", "Mordekaiserb", "Morganab",
    "Namib", "Nasusb", "Nautilusb", "Neekob", "Nidaleeb", "Nocturneb", "Nunu & Willumpb",
    "Olafb", "Oriannab", "Ornnb",
    "Pantheonb", "Poppyb", "Pykeb",
    "Qiyanab", "Quinnb",
    "Rakanb", "Rammusb", "Rek'Saib", "Renektonb", "Rengarb", "Rivenb", "Rumbleb", "Ryzeb",
    "Sejuanib", "Shacob", "Shenb", "Shyvanab", "Singedb", "Sionb", "Sivirb", "Skarnerb", "Sonab", "Sorakab", "Swainb", "Sylasb", "Syndrab",
    "Tahm Kenchb", "Taliyahb", "Talonb", "Taricb", "Teemob", "Threshb", "Tristanab", "Trundleb", "Tryndamereb", "Twisted Fateb", "Twitchb",
    "Udyrb", "Urgotb",
    "Varusb", "Vayneb", "Veigarb", "Vel'Kozb", "Vib", "Viktorb", "Vladimirb", "Volibearb",
    "Warwickb", "Wukongb",
    "Xayahb", "Xerathb", "Xin Zhaob",
    "Yasuob", "Yorickb", "Yuumib",
    "Zacb", "Zedb", "Ziggsb", "Zileanb", "Zoeb", "Zyrab",
    "gamelength",
    "playoff",
    "firstblood",
    "firstdragon",
    "firstherald",
    "firstbaron",
    "firsttower",
    "GDa10",
    "XPDa10",
    "GDa15",
    "XPDa15",
    "result"
]

def parseData(file, df):
    # Read Data
    print(file)
    input = pd.read_csv(file, na_values="", dtype = {"url": str})
    input.fillna("", inplace=True)

    # Initialization
    value = {}
    progress_bar = tqdm(total=input.shape[0])
    
    # Initialize First Instance
    game = input["gameid"][0]

    for idx, row in input.iterrows():

        # Update Instance
        if game != row["gameid"]:
            entry = pd.DataFrame([value], index=[0], columns=keys)
            entry.fillna(0, inplace=True)
            df = pd.concat([df, entry])
            value = {}
            game = row["gameid"]
        
        # Update Champion Picks
        elif row["position"] != 'team':
            side = '1' if row['side'] == 'Blue' else '2'
            value[row["champion"] + side] = 1
        
        # Update Team Stats
        else:
            side = '1' if row['side'] == 'Blue' else '2'
            value['teamname' + side] = row['teamname']
            for i in range(5):
                if row['ban' + str(i + 1)]:
                    value[row['ban' + str(i + 1)] + 'b'] = 1
            value['kills' + side] =     row['kills']
            value['assists' + side] =   row['assists']
            value['TKPM' + side] =      row['team kpm']
            value['CKPM' + side] =      row['ckpm']
            value["infernal" + side] =  row['infernals']
            value["mountain" + side] =  row['mountains']
            value["cloud"] =            row['clouds']
            value["ocean" + side] =     row['oceans']
            value["chemtech" + side] =  row['chemtechs']
            value["hextech" + side] =   row['hextechs'] 
            value["elders" + side] =    row['elders']
            value["heralds" + side] =   row['heralds']
            value["barons" + side] =    row['barons']
            value["towers" + side] =    row['towers']
            value["turrets" + side] =   row['turretplates']
            value["inhibitor" + side] = row['inhibitors']
            value["damage" + side] =    row['damagetochampions']
            value["DPM" + side] =       row['dpm']
            value["DTPM" + side] =      row['damagetakenperminute']
            value["DMPM" + side] =      row['damagemitigatedperminute']
            value["wards" + side] =     row['wardsplaced']
            value["WPM" + side] =       row['wpm']
            value["WK" + side] =        row['wardskilled']
            value["WKPM" + side] =      row['wcpm']
            value["CW" + side] =        row['controlwardsbought']
            value["VS" + side] =        row['visionscore']
            value["VSPM" + side] =      row['vspm']
            value["Gold" + side] =      row['totalgold']
            value["EG" + side] =        row['earnedgold']
            value["EGPM" + side] =      row['earned gpm']
            value["GS" + side] =        row['goldspent']
            value["GSPD" + side] =      row['gspd']
            value["MK" + side] =        row['monsterkills']
            value["Ga10" + side] =      row['goldat10']
            value["XPa10" + side] =     row['xpat10']
            value["Ka10" + side] =      row['killsat10']
            value["Aa10" + side] =      row['assistsat10']
            value["Ga15" + side] =      row['goldat15']
            value["XPa15" + side] =     row['xpat15']
            value["Ka15" + side] =      row['killsat15']
            value["Aa15" + side] =      row['assistsat15']

            # Update Shared Stats
            if side == '2':
                value["gamelength"] =   row['gamelength']
                value["playoff"] =      row['playoffs']
                value["firstblood"] =   row['firstblood']
                value["firstdragon"] =  row['firstdragon']
                value["firstherald"] =  row['firstherald']
                value["firstbaron"] =   row['firstbaron']
                value["firsttower"] =   row['firsttower']
                value["GDa10"] =        row['golddiffat10']
                value["XPDa10"] =       row['xpdiffat10']
                value["GDa15"] =        row['golddiffat15']
                value["XPDa15"] =       row['xpdiffat15']
                value["result"] =       row['result']
        
        progress_bar.update(1)
    
    # Update Last Instance
    entry = pd.DataFrame([value], index=[0], columns=keys)
    entry.fillna(0, inplace=True)
    df = pd.concat([df, entry])

    progress_bar.close()
    
    return df

if __name__ == '__main__':
    df = pd.DataFrame(columns=keys)
    df = parseData('2017_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('A_data_2017.csv', index=False)
    df = parseData('2018_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('A_data_2018.csv', index=False)
    df = parseData('2019_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('A_data_2019.csv', index=False)
    df = parseData('2020_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('A_data_2020.csv', index=False)
    df = parseData('2021_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('A_data_2021.csv', index=False)
    df = parseData('2022_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('A_data_2022.csv', index=False)
