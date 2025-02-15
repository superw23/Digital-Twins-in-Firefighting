using UnityEngine;
using UnityEngine.UI;
using System.Collections;

using TMPro;
public class TimerScript : MonoBehaviour
{
    public TextMeshProUGUI timerText;
    private float startTime;
    private bool isRunning = false;

    void Start()
    {
        // 初始化计时器
        startTime = Time.time;
        isRunning = true;
    }

    void Update()
    {
        if (isRunning)
        {
            float t = Time.time - startTime;

            string minutes = ((int)t / 60).ToString("00");
            string seconds = (t % 60).ToString("00");

            timerText.text = minutes + ":" + seconds;
        }
    }

    // 调用这个方法来停止计时器
    public void StopTimer()
    {
        isRunning = false;
    }
}
