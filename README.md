# Anticipation in Football

**From predicting opponents to understanding teammates.**

Anticipation is what lets a player act before a situation fully unfolds, whether by reading an opponent or by knowing where a teammate is about to run. It is usually measured in isolated perceptual tasks, which says little about how it shows up continuously during a match. **Anticipation maps** make it observable from player-tracking data: for every pair of players, they count the match phases in which one player's movement statistically *leads* the other's, after controlling for the ball. Teammate-to-teammate and player-to-opponent anticipation are kept separate.

This repository accompanies the **SSAC 2027 abstract** *"Anticipation in Football: From Predicting Opponents to Understanding Teammates"*.

## How it works

Built on the events and player-tracking data of the [PFF FC 2022 World Cup dataset](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data).

1. **Movement profiles.** For each event, we extract every player's directional velocity, speed and distance to the ball over a window running from shortly before the event to the next event. The ball-motion profile is extracted over the same window.
2. **Granger causality.** We test pairs of player profiles for significant directional temporal dependencies, using the ball-motion profile as a control so that movement driven by the ball is not mistaken for anticipation. This is applied to both intra-team and inter-team pairs.
3. **Recurrence maps.** We build maps of how these predictive relationships are structured across the game, separately for teammate-to-teammate and player-to-opponent anticipation.

## What the evidence shows

Anticipation maps reveal distinct tactical hierarchies in teams with different playing styles.

| Team (match) | Strongest anticipation link(s) | Phases |
| --- | --- | --- |
| France (Final vs Argentina) | Tchouaméni ↔ Rabiot, reciprocal | 293 / 275 |
| Argentina (Semi-final 3–0 vs Croatia) | Mac Allister ↔ Messi, reciprocal | 218 |
| | Álvarez → Messi, into vacated attacking pockets | 169 |
| Morocco (Semi-final vs France) | Ziyech ↔ Hakimi, near-perfect reciprocity | 187 / 183 |
| Serbia (Group stage 3–3 vs Cameroon) | Tadić → Mitrović, box runs | 215 / 206 |

Each count is the number of 15-second tactical phases in which the leader (row) Granger-caused the follower (column) at p < 0.05, conditioning on the ball trajectory.

In the **2022 World Cup Final**, separating within-team from cross-team anticipation shows:

- **France in-team:** Rabiot → Tchouaméni leads (293).
- **Argentina reading France:** Mac Allister → Tchouaméni leads (320).
- **France reading Argentina:** Tchouaméni → Mac Allister (308), Messi (287) and Álvarez (274).

There is no standard reference measure of anticipation, so results are compared against published literature and expert tactical analyses (FIFA Training Centre, Total Football Analysis, Coaches' Voice), which document unusually strong on-field coordination during the tournament. Players who entered late in the Final (Lautaro Martínez, 102nd minute; Dybala, 120+1) show near-zero values by construction.

## Applications

Evaluating player intelligence, team coordination, scouting and tactical analysis.
