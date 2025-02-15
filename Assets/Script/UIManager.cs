
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class UIManger : MonoBehaviour
{
    [SerializeField] private TMP_Text windValue;
    [SerializeField] private TMP_Text tempValue;
    [SerializeField] private TMP_Text FSIValue;
    [SerializeField] private Slider windSlider;
    [SerializeField] private Slider tempSlider;
    [SerializeField] private Slider fsiSlider;
    [SerializeField] private string nextSceneName = "HouseScene1";
    public Button startButton;

    // Start is called before the first frame update
    void Start()
    {
        windSlider.onValueChanged.AddListener(delegate { UpdateWindValue(); });
        tempSlider.onValueChanged.AddListener(delegate { UpdateTempValue(); });
        fsiSlider.onValueChanged.AddListener(delegate { UpdateFSIValue(); });
        startButton.onClick.AddListener(StartGame);
    }
    
    void StartGame()
    {
        // save the value for the main scene
        PlayerPrefs.SetInt("WindValue", Mathf.RoundToInt(windSlider.value));
        PlayerPrefs.SetInt("TempValue", Mathf.RoundToInt(tempSlider.value));
        PlayerPrefs.SetInt("FSIValue", Mathf.RoundToInt(fsiSlider.value));
        //load the main scene
        SceneManager.LoadScene("MainScene");
    }
    
    public void UpdateWindValue()
    {
        windValue.text = windSlider.value.ToString("0");
    }
    
    public void UpdateFSIValue()
    {
        FSIValue.text = fsiSlider.value.ToString("0");
    }
    
    public void UpdateTempValue()
    {
        tempValue.text = tempSlider.value.ToString("0");
    }
}
