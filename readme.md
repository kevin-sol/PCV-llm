$$
QoE_{frame}=A·QoE_{quality}-B·QoE_{rebuffer}-C·QoE_{switch}\\
QoE_{quality}=quality\_level·\sum_{frame}\sum_{tile}fov_{predicted}[frame][tile]·fov_{real}[frame][tile]\\
QoE_{rebuffer}=rebuffer\_time\\
QoE_{switch}=|quality\_level_{last\_frame}-quality\_level_{current\_frame}|
$$

$$
A=50\\
B=3000\\
C=5
$$

运行训练

```
python --train
```
