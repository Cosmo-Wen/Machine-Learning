import pandas as pd
import json
from tqdm import tqdm

keys = [
    'date',
    'team1',
    'winrate1',
    'totalgame1',
    'killspg1',         # Kills per Game
    'assistspg1',       # Assists per Game
    'deathpg1',         # Deaths per Game
    'ckpg1',            # Combined Kills per Game
    'dragonspg1',       # Dragons per Game
    'towerspg1',        # Towers per Game
    'dmgpg1',           # Damage per Game
    'visionscorepg1',   # Vision Score per Game
    'goldpg1',          # Gold per Game
    'CSpg1',            # Creep Score per Game
    'team2',
    'winrate2',
    'totalgame2',
    'killspg2',         # Kills per Game
    'assistspg2',       # Assists per Game
    'deathpg2',         # Deaths per Game
    'ckpg2',            # Combined Kills per Game
    'dragonspg2',       # Dragons per Game
    'towerspg2',        # Towers per Game
    'dmgpg2',           # Damage per Game
    'visionscorepg2',   # Vision Score per Game
    'goldpg2',          # Gold per Game
    'CSpg2',            # Creep Score per Game
    'result'
]

"""
Teams:
    start           - <bool>
    Total Game      - <int>
    Wins            - <int>
    Kills           - <int>
    Assists         - <int>
    Deaths          - <int>
    Combined Kills  - <int>
    Dragons         - <int>
    Towers          - <int>
    Damage          - <int>
    Vision Score    - <int>
    Gold            - <int>
    Creep Score     - <int>
"""

teams = {}

def parseData(file, df):
    # Read Data
    print(file)
    input = pd.read_csv(file, na_values="", dtype = {"url": str})
    input.fillna("", inplace=True)

    # Initialization
    value = {}
    progress_bar = tqdm(total=input.shape[0])
    skip = False
    
    # Initialize First Instance
    game = input["gameid"][0]
    
    for idx, row in input.iterrows():
        
        # Update Instance
        if game != row["gameid"]:
            if not skip:
                entry = pd.DataFrame([value], index=[0], columns=keys)
                entry.fillna(0, inplace=True)
                df = pd.concat([df, entry])
            value = {}
            skip = False    
            game = row["gameid"]

        elif skip: continue
        
        # Skip 'unknown teams'
        elif row['teamid'] == '': 
            skip = True
            continue

        # Update Team Record
        elif row['position'] == 'team':
            # Initialize Team
            if not row['teamid'] in teams:
                teams[row['teamid']] = {'start': True,
                                        'games':    0, 
                                        'wins':     0, 
                                        'kills':    0,
                                        'assists':  0,
                                        'deaths':   0, 
                                        'CK':       0, 
                                        'dragons':  0,
                                        'towers':   0,
                                        'damage':   0,
                                        'VS':       0,
                                        'gold':     0,
                                        'CS':       0,
                                        }

            # Place values in the temporary dataframe
            side = '1' if row['side'] == 'Blue' else '2'
            teamid = row['teamid']
            value['date'] =                 row['date']
            value['team' + side] =          teamid
            value['winrate' + side] =       0 if teams[teamid]['start'] else teams[teamid]['wins'] / teams[teamid]['games']
            value['totalgame' + side] =     0 if teams[teamid]['start'] else teams[teamid]['games']
            value['killspg' + side] =       0 if teams[teamid]['start'] else teams[teamid]['kills'] / teams[teamid]['games']
            value['assistspg' + side] =     0 if teams[teamid]['start'] else teams[teamid]['assists'] / teams[teamid]['games']
            value['deathpg' + side] =       0 if teams[teamid]['start'] else teams[teamid]['deaths'] / teams[teamid]['games']
            value['ckpg' + side] =          0 if teams[teamid]['start'] else teams[teamid]['CK'] / teams[teamid]['games']
            value['dragonspg' + side] =     0 if teams[teamid]['start'] else teams[teamid]['dragons'] / teams[teamid]['games']
            value['towerspg' + side] =      0 if teams[teamid]['start'] else teams[teamid]['towers'] / teams[teamid]['games']
            value['dmgpg' + side] =         0 if teams[teamid]['start'] else teams[teamid]['damage'] / teams[teamid]['games']
            value['visionscorepg' + side] = 0 if teams[teamid]['start'] else teams[teamid]['VS'] / teams[teamid]['games']
            value['goldpg' + side] =        0 if teams[teamid]['start'] else teams[teamid]['gold'] / teams[teamid]['games']
            value['CSpg' + side] =          0 if teams[teamid]['start'] else teams[teamid]['CS'] / teams[teamid]['games']
            if side == '2': 
                value['result'] =           row['result']

            # Update total statistics
            teams[teamid]['start'] =    False
            teams[teamid]['games'] +=   1
            teams[teamid]['wins'] +=    int(row['result'])
            teams[teamid]['kills'] +=   int(0 if row['kills'] == '' else int(row['kills']))
            teams[teamid]['assists'] += int(0 if row['assists'] == '' else int(row['assists']))
            teams[teamid]['deaths'] +=  int(0 if row['deaths'] == '' else int(row['deaths']))
            teams[teamid]['CK'] +=      int(0 if row['kills'] == '' or row['deaths'] == '' else int(row['kills']) - int(row['deaths']))
            teams[teamid]['dragons'] += int(0 if row['dragons'] == '' else int(row['dragons']))
            teams[teamid]['towers'] +=  int(0 if row['towers'] == '' else int(row['towers']))
            teams[teamid]['damage'] +=  int(0 if row['damagetochampions'] == '' else int(row['damagetochampions']))
            teams[teamid]['VS'] +=      int(0 if row['visionscore'] == '' else int(row['visionscore']))
            teams[teamid]['gold'] +=    int(0 if row['totalgold'] == '' else int(row['totalgold']))
            teams[teamid]['CS'] +=      int(0 if row['total cs'] == '' else int(row['total cs']))

        progress_bar.update(1)
    
    # Update Last Instance
    if not skip:
        entry = pd.DataFrame([value], index=[0], columns=keys)
        entry.fillna(0, inplace=True)
        df = pd.concat([df, entry])

    progress_bar.close()
    
    return df

def writeTeams(filename):
    global teams
    # Convert the dictionary to a JSON string
    json_str = json.dumps(teams, indent=2)

    # Open the .txt file in write mode
    with open(filename, 'w') as f:
        # Write the JSON string to the .txt file
        f.write(json_str)
    teams = {}

if __name__ == '__main__':
    df = pd.DataFrame(columns=keys)
    df = parseData('2017_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('B_data_2017.csv', index=False)
    writeTeams('B_data_2017_teams.txt')
    df = parseData('2018_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('B_data_2018.csv', index=False)
    writeTeams('B_data_2018_teams.txt')
    df = parseData('2019_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('B_data_2019.csv', index=False)
    writeTeams('B_data_2019_teams.txt')
    df = parseData('2020_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('B_data_2020.csv', index=False)
    writeTeams('B_data_2020_teams.txt')
    df = parseData('2021_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('B_data_2021.csv', index=False)
    writeTeams('B_data_2021_teams.txt')
    df = parseData('2022_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('B_data_2022.csv', index=False)
    writeTeams('B_data_2022_teams.txt')
    df = parseData('2017_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df = parseData('2018_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df = parseData('2019_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df = parseData('2020_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df = parseData('2021_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df = parseData('2022_LoL_esports_match_data_from_OraclesElixir.csv', df)
    df.to_csv('B_data.csv', index=False)
    writeTeams('B_data_teams.txt')
