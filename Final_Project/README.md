# Machine-Learning-Final

This includes the dataset and models for the final project of machine learning 2022.

There are three sections of evaluations, please refer to the info below :

# How to use:

## Part A:

Part A is the information of each game in the data collection. To use the data, consider each game as independent, and proceed to train with instances.

Part A consists of the following data:

==== Team 1 ====

- Team Name1 - < string >
- Champion1 - < bool[num of champions] > (Total 5 True)
- Kills1 - < int >
- Assists1 - < int >
- Total Kills per Minute 1 - < float >
- Combined Kills per Minute 1 - < float >
- Infernal Dragon1 - < int >
- Mountain Dragon1 - < int >
- Cloud Dragon1 - < int >
- Ocean Dragon1 - < int >
- Chemtech Dragon1 - < int >
- Hextech Dragon1 - < int >
- Elders1 - < int >
- heralds1 - < int >
- Barons1 - < int >
- Towers1 - < int >
- Turrets1 - < int >
- Inhibitors1 - < int >
- Total Damage1 - < int >
- Damage per Minute1 - < float >
- Damage Taken per Minute1 - < float >
- Damage Mitigated per Minute1 - < float >
- Wards1 - < int >
- Wards per Minute1 - < float >
- Wards Killed1 - < int >
- Wards Killed per Minute1 - < float >
- Control Wards1 - < int >
- Vision Score1 - < int >
- Visual Score per Minute1 - < int >
- Total Gold1 - < int >
- Earned Gold1 - < int >
- Earned Gold per Minute1 - < float >
- Gold Spent1 - < int >
- Gold Spent per Minute1 - < float >
- Monsters Killed1 - < int >
- Gold at 10 1 - < int >
- XP at 10 1 - < int >
- Kills at 10 1 - < int >
- Assists at 10 1 - < int >
- Gold at 15 1 - < int >
- XP at 15 1 - < int >
- Kills at 15 1 - < int >
- Assists at 15 1 - < int >

==== Team 2 ====

- Team Name2 - < string >
- Champion2 - < bool[num of champions] > (Total 5 True)
- Kills2 < int >
- Assists2 < int >
- Total Kills per Minute 2 < float >
- Combined Kills per Minute 2 < float >
- Infernal Dragon2 - < int >
- Mountain Dragon2 - < int >
- Cloud Dragon2 - < int >
- Ocean Dragon2 - < int >
- Chemtech Dragon2 - < int >
- Hextech Dragon2 - < int >
- Elders2 - < int >
- herald2 - < int >
- Baron2 - < int >
- Towers2 - < int >
- Turrets2 - < int >
- Inhibitors2 - < int >
- Total Damage2 - < int >
- Damage per Minute2 - < float >
- Damage Taken per Minute2 - < float >
- Damage Mitigated per Minute2 - < float >
- Wards2 - < int >
- Wards per Minute2 - < float >
- Wards Killed2 - < int >
- Wards Killed per Minute2 - < float >
- Control Wards2 - < int >
- Vision Score2 - < int >
- Visual Score per Minute2 < int >
- Total Gold2 - < int >
- Earned Gold2 - < int >
- Earned Gold per Minute2 - < float >
- Gold Spent2 - < int >
- Gold Spent per Minute2 - < float >
- Monsters Killed2 - < int >
- Gold at 10 2 - < int >
- XP at 10 2 - < int >
- Kills at 10 2 - < int >
- Assists at 10 2 - < int >
- Gold at 15 2 - < int >
- XP at 15 2 - < int >
- Kills at 15 2 - < int >
- Assists at 15 2 - < int >

==== Shared Stats ====

- Bans - < bool[num of champions] > (Total of 10 True)
- Game Length - < int >
- Playoff - < bool >
- First Blood - < bool > (0 for Team 1, 1 for Team 2)
- First Dragon - < bool > (0 for Team 1, 1 for Team 2)
- First herald - < bool > (0 for Team 1, 1 for Team 2)
- First Baron - < bool > (0 for Team 1, 1 for Team 2)
- First Tower - < bool > (0 for Team 1, 1 for Team 2)
- Gold Diff at 15 - < int > (Negative for Team 1, Positive for Team 2)
- XP Diff at 15 - < int > (Negative for Team 1, Positive for Team 2)
- Gold Diff at 15 - < int > (Negative for Team 1, Positive for Team 2)
- XP Diff at 15 - < int > (Negative for Team 1, Positive for Team 2)
- result - < bool > (0 for Team 1, 1 for Team 2)

## Part B:

Part B is the moving average of specific stats of a team. Each instance is a game. The labels in each game statistic are the previous records of both teams (not including the current game). This imitates knowing the track record of every team before the game has occurred.

Part B consists of the following data:

In Data:

==== Game Info ====

- Date - < string >

==== Team 1 ====

- Team ID 1 - < string >
- Win Rate 1 - < float >
- Total Game Count 1 - < int >
- Kills per Game 1 - < float >
- Assists per Game 1 - < float >
- Deaths per Game 1 - < float >
- Combined Kills per Game 1 - < float >
- Dragons per Game 1 - < float >
- Towers per Game 1 - < float >
- Damage per Game 1 - < float >
- Vision Score per Game 1 - < float >
- Gold per Game 1 - < float >
- Creep Score per Game 1 - < float >

==== Team 2 ====

- Team ID 2 - < string >
- Win Rate 2 - < float >
- Total Game Count 2 - < int >
- Kills per Game 2 - < float >
- Assists per Game 2 - < float >
- Deaths per Game 2 - < float >
- Combined Kills per Game 1 - < float >
- Dragons per Game 2 - < float >
- Towers per Game 2 - < float >
- Damage per Game 2 - < float >
- Vision Score per Game 2 - < float >
- Gold per Game 2 - < float >
- Creep Score per Game 2 - < float >

==== Result ====

- Result - < bool > (0 for Team 1, 1 for Team 2)

In Data_Teams:

==== Teams ====

- start - < bool > (Dummy Variable)
- Total Game - < int >
- Wins - < int >
- Kills - < int >
- Assists - < int >
- Deaths - < int >
- Combined Kills - < int >
- Dragons - < int >
- Towers - < int >
- Damage - < int >
- Vision Score - < int >
- Gold - < int >
- Creep Score - < int >

## Part C:

Part C can be derived from the data in Part A. Therefore, there isn't another file containing the data. Just read the data from Part A and use only the following statistics:

==== Team 1 ====

- Team Name1 - < string >
- Champion1 - < bool[num of champions] > (Total 5 True)
- Infernal Dragon1 - < int >
- Mountain Dragon1 - < int >
- Cloud Dragon1 - < int >
- Ocean Dragon1 - < int >
- Chemtech Dragon1 - < int >
- Hextech Dragon1 - < int >
- Gold at 10 1 - < int >
- XP at 10 1 - < int >
- Kills at 10 1 - < int >
- Assists at 10 1 - < int >
- Gold at 15 1 - < int >
- XP at 15 1 - < int >
- Kills at 15 1 - < int >
- Assists at 15 1 - < int >

==== Team 2 ====

- Team Name2 - < string >
- Champion2 - < bool[num of champions] > (Total 5 True)
- Infernal Dragon2 - < int >
- Mountain Dragon2 - < int >
- Cloud Dragon2 - < int >
- Ocean Dragon2 - < int >
- Chemtech Dragon2 - < int >
- Hextech Dragon2 - < int >
- Gold at 10 2 - < int >
- XP at 10 2 - < int >
- Kills at 10 2 - < int >
- Assists at 10 2 - < int >
- Gold at 15 2 - < int >
- XP at 15 2 - < int >
- Kills at 15 2 - < int >
- Assists at 15 2 - < int >

==== Shared Stats ====

- Bans - < bool[num of champions] > (Total of 10 True)
- First Blood - < bool > (0 for Team 1, 1 for Team 2)
- First Dragon - < bool > (0 for Team 1, 1 for Team 2)
- First herald - < bool > (0 for Team 1, 1 for Team 2)
- First Baron - < bool > (0 for Team 1, 1 for Team 2)
- First Tower - < bool > (0 for Team 1, 1 for Team 2)
- Gold Diff at 15 - < int > (Negative for Team 1, Positive for Team 2)
- XP Diff at 15 - < int > (Negative for Team 1, Positive for Team 2)
- Gold Diff at 15 - < int > (Negative for Team 1, Positive for Team 2)
- XP Diff at 15 - < int > (Negative for Team 1, Positive for Team 2)
- result - < bool > (0 for Team 1, 1 for Team 2)

# Models
The models are named _ part evaluation 20__ .ipynb

Used classification models are KNN, SVC, voting (LR DT KNN), and bagging with decision tree.
