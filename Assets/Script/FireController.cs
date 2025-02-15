using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using System.Text;
using System.Collections.Generic;
using TMPro;
using Unity.VisualScripting;
using UnityEngine.SceneManagement;
using UnityEngine.Serialization;
using UnityEngine.UI;
using Random = UnityEngine.Random;

// Serializable class to hold fire data for JSON serialization.
[System.Serializable]
public class FireData
{
    public int Temperature;
    public int Wind_Speed;
    public int FFMC;
    public int DMC;
    public int ISI;
}

[System.Serializable]
public class FireDataListContainer
{
    public List<FireData> data;
}

// Serializable class to hold probabilities after Flask server response.
[System.Serializable]
public class ProbabilityList
{
    public List<float> probabilities;
}

// Main controller for the fire prediction and visualization.
public class FireController : MonoBehaviour
{
    public GameObject playerCharacter; // Assign your character GameObject in the Inspector
    public GameObject loadingIndicator; // initialize the loading UI

    // Variables for simulation parameters and the URL of the Flask server.
    private int windSpeed;
    private int temperature;
    private string flaskUrl = "http://localhost:5000/predictFire";
    
    // Variables for grid dimensions and scene bounds.
    public int gridWidth = 100;
    public int gridHeight = 100;
    public int sceneMaxX = 66;
    public int sceneMinX = 49;
    public int sceneMaxY = 57;
    public int sceneMinY = 49;

    // Arrays to hold the current state of each grid cell and its risk level.
    public int[,] gridState1;
    public int[,] riskGrid1;
    public int[,] gridState2;
    public int[,] riskGrid2;
    public int[,] gridState3;
    public int[,] riskGrid3;

    public int[,] floorRiskGrid1;
    public int[,] floorRiskGrid2;
    public int[,] floorRiskGrid3;
    
    public int currentFloor = 0;
    public int currentRiskFloor = 0;

    // Variables for controlling the flow and timing of updates.
    private float timeSinceLastUpdate = 0.0f;
    private List<Vector2Int> newFires;
    public float updateInterval = 3.0f; // time interval for fire update
    
    public GameObject pathPrefab;
    
    // Positions for starting points, exits, and mid-points on different floors.
    public Vector2Int startPosition = new Vector2Int(62, 50);
    public Vector2Int exitPosition = new Vector2Int(59, 56);
    public Vector2Int floor1MiddlePosition = new Vector2Int(59, 50);
    public Vector2Int floor2MiddlePosition = new Vector2Int(53, 57);
    public Vector2Int floor3MiddlePosition = new Vector2Int(54, 56);
    public Vector2Int stairPosition = new Vector2Int(51, 50);

    // GameObject references to various fire and smoke effects.
    public GameObject largeFlames;
    public GameObject mediumFlames;
    public GameObject tinyFlames;
    
    public GameObject greySmoke;
    public GameObject blackSmoke;
    public GameObject whiteSmoke;
    public GameObject smallSmoke;
    public GameObject largeSmoke;

    public GameObject vfxSmallFire;
    public GameObject vfxMediumFire;
    public GameObject vfxBigFire;
    public int windValue;
    public int tempValue;
    public int FSIValue;
    public float currentRiskMultifier;

    // Boolean Values
    public bool isPrediction = false;
    public bool isRiskGridUpdated1 = false;
    public bool isRiskGridUpdated2 = false;
    public bool isRiskGridUpdated3 = false;
    public int processResponseTime = 0;
    
    // represent the risk level in the scene
    public GameObject blockPrefab; 
    public GameObject yellowBlock;
    public GameObject orangeBlock;
    public GameObject redBlock;
    
    // Variables to track the player's health and update UI elements.
    private int contactWithTinyFlame = 0;
    private int contactWithMediumFlame = 0;
    private int contactWithLargeFlame = 0;
    
    // References to UI elements and text for displaying information
    public TextMeshProUGUI healthText;
    public int health = 250;
    public TextMeshProUGUI guideText;
    public int guideTime = 0;
    [SerializeField] private TMP_Text windText;
    [SerializeField] private TMP_Text tempText;
    [SerializeField] private TMP_Text FSIText;
    [SerializeField] private Slider windSlider;
    [SerializeField] private Slider tempSlider;
    [SerializeField] private Slider fsiSlider;
    
    
    // Updates health display based on player's interactions with fire.
    private void UpdateHealthDisplay()
    {
        health = 250 - contactWithLargeFlame * 5 - contactWithMediumFlame * 2 - contactWithTinyFlame * 1;
        if (health<=0)
        {
            health = 0;
        }
        healthText.text = $"Health Value: {health}/250";
    }
    
    // Updates the display for the number of times the evacuation route has been updated.
    private void UpdateGuideTimeDisplay()
    {
        guideText.text = $"The Guide Route is Updated: {guideTime} Times";
    }

    // Increments the count of tiny flame contacts which will affect the player's health.
    public void IncreaseContactWithTinyFlame()
    {
        contactWithTinyFlame++;
    }
    public void IncreaseContactWithMediumFlame()
    {
        contactWithMediumFlame++;
    }
    public void IncreaseContactWithLargeFlame()
    {
        contactWithLargeFlame++;
    }

    // Initial setup called before the first frame update.
    void Start()
    {
        UpdateHealthDisplay();
        UpdateGuideTimeDisplay();
        // initialize the User-defined parameters
        InitializeUI();
        AddListenertoSlider();
        Debug.Log(FSIValue.ToString()+"  "+tempValue.ToString()+"  "+windValue.ToString());
        // set a loading sign in scene
        loadingIndicator.SetActive(true);
        // Set the character position
        SetCharacterPosition(64, 7.2f, 50); // floor 3
        //SetCharacterPosition(62, 4.05f, 50); // floor 2
        //SetCharacterPosition(62, 1.15f, 50); // floor 1
        
        // initialize the fire grid
        InitializeGrid();
        gridState1[53, 55] = 1;
        gridState2[53, 55] = 1;
        gridState3[53, 55] = 1;
        StartCoroutine(SendFireDataToFlask(riskGrid1));
        StartCoroutine(SendFireDataToFlask(riskGrid2));
        StartCoroutine(SendFireDataToFlask(riskGrid3));
        isPrediction = true;
        
        

        // make sure to render the suggested path after Python Prediction is done
        StartCoroutine(WaitForRiskGridAndUpdatePath());
        // Call UpdatePathMethod every `updateInterval` seconds, starting after `updateInterval` seconds.
        InvokeRepeating("UpdatePathMethod", 5.0f, 2.0f);
    }
    
    // new position will be set based on the grid, plus an offset if needed.
    public void SetCharacterPosition(float x, float y, float z)
    {
        Vector3 newPosition = new Vector3(x, y, z);
        playerCharacter.transform.position = newPosition;
    }

    // Waits for the risk grid update before calculating and displaying an evacuation path.
    IEnumerator WaitForRiskGridAndUpdatePath()
    {
        // wait for riskGrid is updated
        yield return new WaitUntil(() => isRiskGridUpdated1);
        currentRiskFloor = 1;
        // find and display the path
        FindAndDisplayPath(riskGrid1,gridState1,stairPosition,floor1MiddlePosition);
        FindAndDisplayPath(riskGrid1,gridState1,exitPosition,floor1MiddlePosition);
        //FindAndDisplayPath(riskGrid1,stairPosition,exitPosition);
        RenderRiskGrid(riskGrid1,currentRiskFloor);
        
        // wait for riskGrid is updated
        yield return new WaitUntil(() => isRiskGridUpdated2);
        currentRiskFloor = 2;
        // find and display the path
        RenderRiskGrid(riskGrid2,currentRiskFloor);
        
        
        // wait for riskGrid is updated
        yield return new WaitUntil(() => isRiskGridUpdated3);
        currentRiskFloor = 3;
        // find and display the path
        FindAndDisplayPath(riskGrid3,gridState3,startPosition,floor3MiddlePosition);
        FindAndDisplayPath(riskGrid3,gridState3,floor3MiddlePosition,stairPosition);
        RenderRiskGrid(riskGrid3,currentRiskFloor);
    }
    
    // Regularly updates the evacuation path based on the risk grid.
    void UpdatePathMethod()
    {
        // Clear old signs before generating new path.
        DeleteSign(); 
        Debug.Log("update path");
        guideTime++;
        //initialize a new pathPrefab
        pathPrefab=Instantiate(pathPrefab, new Vector3(0, 0, 0), Quaternion.identity);
        currentRiskFloor = 1;
        // find and display the path
        FindAndDisplayPath(riskGrid1,gridState1,stairPosition,floor1MiddlePosition);
        FindAndDisplayPath(riskGrid1,gridState1,exitPosition,floor1MiddlePosition);
        //FindAndDisplayPath(riskGrid1,stairPosition,exitPosition);
        
        // wait for riskGrid is updated
        currentRiskFloor = 2;
        // find and display the path
        
        // wait for riskGrid is updated
        currentRiskFloor = 3;
        // find and display the path
        FindAndDisplayPath(riskGrid3,gridState3,startPosition,floor3MiddlePosition);
        FindAndDisplayPath(riskGrid3,gridState3,floor3MiddlePosition,stairPosition);
    }

    // Initializes the state of the grid representing the game world.
    void InitializeGrid()
    {
        gridState1 = new int[gridWidth, gridHeight];
        riskGrid1 = new int[gridWidth, gridHeight];
        gridState2 = new int[gridWidth, gridHeight];
        riskGrid2 = new int[gridWidth, gridHeight];
        gridState3 = new int[gridWidth, gridHeight];
        riskGrid3 = new int[gridWidth, gridHeight];
        floorRiskGrid1=new int[gridWidth, gridHeight];
        floorRiskGrid2=new int[gridWidth, gridHeight];
        floorRiskGrid3=new int[gridWidth, gridHeight];
        for (int x = 0; x < gridWidth; x++)
        {
            for (int y = 0; y < gridHeight; y++)
            {
                gridState1[x, y] = 0; // initialize all the cells to state with no fire
                riskGrid1[x, y] = 0; // initialize risk level 0
                gridState2[x, y] = 0; // initialize all the cells to state with no fire
                riskGrid2[x, y] = 0; // initialize risk level 0
                gridState3[x, y] = 0; // initialize all the cells to state with no fire
                riskGrid3[x, y] = 0; // initialize risk level 0
            }
        }
    }

    // Regular update method called once per frame to handle simulation updates.
    void Update()
    {
        timeSinceLastUpdate += Time.deltaTime;
        if (timeSinceLastUpdate >= updateInterval)
        {
            if (isPrediction)
            {
                currentRiskMultifier = CalculateFireRiskLevel(windValue, tempValue, FSIValue);
                // update for UI elements
                UpdateGuideTimeDisplay();
                UpdateHealthDisplay();
                currentFloor = 1;
                UpdateGridStateForFloor(gridState1, riskGrid1);
                currentFloor = 2;
                UpdateGridStateForFloor2(gridState2, riskGrid2);
                currentFloor = 3;
                UpdateGridStateForFloor3(gridState3, riskGrid3);
            }
            // reset the update time
            timeSinceLastUpdate = 0; 
        }
    }

    // Sends fire data to the Flask server for processing and retrieves predictions.
    IEnumerator SendFireDataToFlask(int[,] riskGrid)
    {
        // Activates the loading indicator to inform the user that processing is underway.
        loadingIndicator.SetActive(true);
        // Initializes a list to hold fire data objects for each grid point in the scene.
        List<FireData> fireDataList = new List<FireData>();
        // Loops through each grid point within the specified scene boundaries.
        for (int x = sceneMinX; x <= sceneMaxX; x++)
        {
            for (int y = sceneMinY; y <= sceneMaxY; y++)
            {
                FireData data = new FireData
                {
                    Temperature = Random.Range(tempValue-5, tempValue+5),
                    Wind_Speed = Random.Range(0, windValue*7),
                    FFMC = Random.Range(5, 100*FSIValue),
                    DMC = Random.Range(0, 290),
                    ISI = Random.Range(0, 20)
                };
                // Adds the newly created fire data instance to the list.
                fireDataList.Add(data);
            }
        }
        // Wraps the list of fire data in a container object for JSON serialization.
        FireDataListContainer container = new FireDataListContainer { data = fireDataList };
        // Converts the fire data container into a JSON formatted string.
        string jsonData = JsonUtility.ToJson(container, true);

        // Initializes a new web request to the Flask server using the PUT method with JSON data.
        using (UnityWebRequest www = UnityWebRequest.Put(flaskUrl, Encoding.UTF8.GetBytes(jsonData)))
        {
            // Specifies that the request method is POST.
            www.method = "POST";
            www.SetRequestHeader("Content-Type", "application/json");

            // Sends the web request and waits for a response.
            yield return www.SendWebRequest();
            // Hides the loading indicator once the web request is done.
            loadingIndicator.SetActive(false);

            // Checks if there was a connection or protocol error during the web request.
            if (www.result == UnityWebRequest.Result.ConnectionError ||
                www.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.LogError("Error: " + www.error);
                Debug.LogError("please connect Flask Server");
            }
            else
            {
                // process the results of prediction
                ProcessResponse(www.downloadHandler.text, riskGrid);
            }
        }
    }

    // Processes the server's JSON response and updates the risk grid accordingly.
    void ProcessResponse(string jsonResponse, int[,] riskGrid)
    {
        ProbabilityList probabilityList =
            JsonUtility.FromJson<ProbabilityList>("{\"probabilities\":" + jsonResponse + "}");
        //Debug.Log("JSON Response: " + "{\"probabilities\":" + jsonResponse + "}");
        if (probabilityList == null || probabilityList.probabilities == null)
        {
            Debug.LogError("Failed to load probability list from JSON.");
        }
        else
        {
            Debug.Log("Loaded probabilities count: " + probabilityList.probabilities.Count);
        }
        int index = 0; 
        for (int x = sceneMinX; x <= sceneMaxX; x++)
        {
            for (int y = sceneMinY; y <= sceneMaxY; y++)
            {
                //check index within probabilities
                if (index < probabilityList.probabilities.Count)
                {
                    float probability = probabilityList.probabilities[index];
                    int riskLevel;
                    float chance = Random.Range(0.0f, 1.0f);
                    if (probability < 16)
                    {
                        riskLevel = 1;
                    }
                    else if (probability < 18)
                    {
                        if (chance < 0.5f) { riskLevel = 1; }
                        riskLevel = 2;
                    }
                    else if (probability < 23)
                    {
                        if (chance < 0.3) { riskLevel = 1; }
                        else if (chance < 0.8f) { riskLevel = 2; }
                        else { riskLevel = 3; }
                    }
                    else
                    {
                        if (chance < 0.15) { riskLevel = 2; }
                        else if (chance < 0.50) { riskLevel = 3; }
                        else { riskLevel = 4; }
                    }
                    riskGrid[x, y] = riskLevel; // set risk in particular cell in riskGrid
                    index++; // move to next cell in Grid
                }
            }
        }
        processResponseTime++;
        isRiskGridUpdated1 = true;
        if (processResponseTime == 2)
        {
            isRiskGridUpdated2 = true;
        }
        if (processResponseTime==3)
        {
            isRiskGridUpdated3 = true;
        }
    }
    
    // Initializes UI elements with default or saved values.
    void InitializeUI()
    {
        // get value from PlayerPrefs
        windValue = PlayerPrefs.GetInt("WindValue", 5);
        tempValue = PlayerPrefs.GetInt("TempValue", 24);
        FSIValue = PlayerPrefs.GetInt("FSIValue", 1);

        // set value for slider
        windSlider.value = windValue;
        tempSlider.value = tempValue;
        fsiSlider.value = FSIValue;

        // display the value
        windText.text = windValue.ToString();
        tempText.text = tempValue.ToString();
        FSIText.text = FSIValue.ToString();
    }
    
    // Attaches listeners to sliders to handle user input and update parameters.
    void AddListenertoSlider()
    {
        windSlider.onValueChanged.AddListener((value) => {
            windText.text = "Wind: " + value.ToString("0");
            //PlayerPrefs.SetInt("WindValue", (int)value);
            windValue = Mathf.RoundToInt(windSlider.value);
        });
        tempSlider.onValueChanged.AddListener((value) => {
            tempText.text = "Temperature: " + value.ToString("0");
            //PlayerPrefs.SetInt("TempValue", (int)value);
            tempValue = Mathf.RoundToInt(tempSlider.value);
        });
        fsiSlider.onValueChanged.AddListener((value) => {
            FSIText.text = "FSI: " + value.ToString("0");
            //PlayerPrefs.SetInt("FSIValue", (int)value);
            FSIValue = Mathf.RoundToInt(fsiSlider.value);
        });
    }
    
    // Renders the risk level on the grid visually for each floor.
    public void RenderRiskGrid(int[,] riskGrid, int riskFloor)
    {
        float riskFloorHeight = .0f;
        switch (riskFloor)
        {
            case 1:
                riskFloorHeight = .095f;
                break;
            case 2:
                riskFloorHeight = 3.08f;
                break;
            case 3:
                riskFloorHeight = 6.08f;
                break;
        }
        for (int x = sceneMinX; x <= sceneMaxX; x++)
        {
            for (int y = sceneMinY; y <= sceneMaxY; y++)
            {
                if ((x>=52&&x<=54)&&(y>=52&&y<=54))
                {
                    continue;
                }
                blockPrefab = null;
                switch (riskGrid[x, y])
                {
                    case 2:
                        blockPrefab = Instantiate(yellowBlock, new Vector3(x, riskFloorHeight, y), Quaternion.identity);
                        break;
                    case 3:
                        blockPrefab = Instantiate(orangeBlock, new Vector3(x, riskFloorHeight, y), Quaternion.identity);
                        break;
                    case 4:
                        blockPrefab = Instantiate(redBlock, new Vector3(x, riskFloorHeight, y), Quaternion.identity);
                        break;
                }
            }
        }
    }
    
    // Updates the state of the grid for a specific floor based on fire spread.
    void UpdateGridStateForFloor(int[,] gridState, int[,] riskGrid)
    {
        int[,] newState = new int[gridWidth, gridHeight];
        // keep record for position to render fire
        List<Vector2Int> newFiresThisUpdate = new List<Vector2Int>(); 

        // make local array
        for (int x = 0; x < gridWidth; x++)
        {
            for (int y = 0; y < gridHeight; y++)
            {
                newState[x, y] = gridState1[x, y];
            }
        }

        // update the grid cell state
        for (int x = sceneMinX; x <= sceneMaxX; x++)
        {
            for (int y = sceneMinY; y <= sceneMaxY; y++)
            {
                // only check cell in fire state 
                if (gridState1[x, y] == 1) 
                {
                    // move to neighbors in four direction
                    List<Vector2Int> directions = new List<Vector2Int>
                    {
                        new Vector2Int(0, 1), // 上
                        new Vector2Int(0, -1), // 下
                        new Vector2Int(1, 0), // 右
                        new Vector2Int(-1, 0) // 左
                    };

                    //keep record for the highest risk level
                    int maxRisk = 0; 
                    Vector2Int maxRiskCell = new Vector2Int(-1, -1); 

                    // move to neighbors in four direction
                    foreach (var dir in directions)
                    {
                        int newX = x + dir.x;
                        int newY = y + dir.y;

                        // make sure it is not fired
                        if (newX >= sceneMinX && newX <= sceneMaxX && newY >= sceneMinY && newY <= sceneMaxY && gridState1[newX, newY] == 0)
                        {
                            if (riskGrid[newX, newY] > maxRisk)
                            {
                                maxRisk = riskGrid[newX, newY];
                                maxRiskCell = new Vector2Int(newX, newY);
                            }
                        }
                    }
                    // if found the cell
                    if (maxRiskCell.x != -1 && maxRiskCell.y != -1)
                    {
                        float chance = Random.Range(0.0f, 1.0f);
                        if (chance < 0.15f*currentRiskMultifier)
                        {
                            newState[maxRiskCell.x, maxRiskCell.y] = 1; 
                            newFiresThisUpdate.Add(maxRiskCell); 
                        }
                    }
                }
            }
        }
        // upgrade the grid
        gridState1 = newState;

        // render fire
        RenderNewFires(newFiresThisUpdate);
    }
    
    // Renders new fire instances on the grid.
    void RenderNewFires(List<Vector2Int> newFires)
    {
        foreach (var fire in newFires)
        {
            Vector3 FirePosition = new Vector3(fire.x, CalculateHeightForFloor(fire), fire.y);
            Vector3 SmokePosition = new Vector3(fire.x, CalculateHeightForFloor(fire) + 2.5f, fire.y);
            GameObject flamePrefab = null;
            bool initializeObj = true;

            // choose diverse flame prefab
            switch (riskGrid1[fire.x, fire.y])
            {
                case 2:
                    flamePrefab = tinyFlames;
                    //flamePrefab = vfxSmallFire;
                    //scaleChange = new Vector3(1.0f, 1.0f, 1.0f); 
                    break;
                case 3:
                    flamePrefab = mediumFlames;
                    //flamePrefab = vfxMediumFire;
                    Instantiate(smallSmoke, FirePosition, Quaternion.Euler(-90, 0, 0));
                    //scaleChange = new Vector3(0.7f, 0.7f, 0.7f);
                    break;
                case 4:
                    flamePrefab = largeFlames;
                    //flamePrefab = vfxBigFire;
                    // render the white smoke
                    Instantiate(largeSmoke, FirePosition, Quaternion.Euler(0, 0, 0));
                    Instantiate(whiteSmoke, FirePosition, Quaternion.Euler(0, 0, 0));

                    //scaleChange = new Vector3(0.7f, 0.7f, 0.7f);
                    break;
                default:
                    // not put fire in scene
                    //Debug.Log("none");
                    initializeObj = false;
                    break;
            }
            if (initializeObj)
            {
                GameObject instance = Instantiate(flamePrefab, FirePosition, Quaternion.Euler(0, 0, 0));
            }
        }
    }

    // Updates grid state for floor 2, considering new fire spread and risk levels.
    void UpdateGridStateForFloor2(int[,] gridState, int[,] riskGrid)
    {
        int[,] newState = new int[gridWidth, gridHeight];
        List<Vector2Int> newFiresThisUpdate = new List<Vector2Int>();

        // copy the array to local variable
        for (int x = 0; x < gridWidth; x++)
        {
            for (int y = 0; y < gridHeight; y++)
            {
                newState[x, y] = gridState2[x, y];
            }
        }
        // iterate with the grid
        for (int x = sceneMinX; x <= sceneMaxX; x++)
        {
            for (int y = sceneMinY; y <= sceneMaxY; y++)
            {
                if (gridState2[x, y] == 1) 
                {
                    // search for 4 directions
                    List<Vector2Int> directions = new List<Vector2Int>
                    {
                        new Vector2Int(0, 1), 
                        new Vector2Int(0, -1), 
                        new Vector2Int(1, 0), 
                        new Vector2Int(-1, 0) 
                    };
                    // find the highest risk
                    int maxRisk = 0; 
                    Vector2Int maxRiskCell = new Vector2Int(-1, -1); 

                    // search for 4 directions
                    foreach (var dir in directions)
                    {
                        int newX = x + dir.x;
                        int newY = y + dir.y;
                        
                        if (newX >= sceneMinX && newX <= sceneMaxX && newY >= sceneMinY && newY <= sceneMaxY && gridState2[newX, newY] == 0)
                        {
                            if (riskGrid[newX, newY] > maxRisk)
                            {
                                maxRisk = riskGrid[newX, newY];
                                maxRiskCell = new Vector2Int(newX, newY);
                            }
                        }
                    }
                    // find the highest risk
                    if (maxRiskCell.x != -1 && maxRiskCell.y != -1)
                    {
                        float chance = Random.Range(0.0f, 1.0f);
                        if (chance < 0.15f*currentRiskMultifier)
                        {
                            newState[maxRiskCell.x, maxRiskCell.y] = 1; 
                            newFiresThisUpdate.Add(maxRiskCell); 
                        }
                    }
                }
            }
        }
        //update state
        gridState2 = newState;
        // render new fires
        RenderNewFires2(newFiresThisUpdate);
    }

// Renders new fire instances on the second floor.
    void RenderNewFires2(List<Vector2Int> newFires)
    {
        foreach (var fire in newFires)
        {
            Vector3 FirePosition = new Vector3(fire.x, CalculateHeightForFloor(fire), fire.y);
            Vector3 SmokePosition = new Vector3(fire.x, CalculateHeightForFloor(fire) + 2.5f, fire.y);
            GameObject flamePrefab = null;
            Vector3 scaleChange = Vector3.one; 
            bool initializeObj = true;

            // choose diverse fire effect
            switch (riskGrid2[fire.x, fire.y])
            {
                case 2:
                    flamePrefab = tinyFlames;
                    //flamePrefab = vfxSmallFire;
                    //scaleChange = new Vector3(1.0f, 1.0f, 1.0f); 
                    break;
                case 3:
                    flamePrefab = mediumFlames;
                    //flamePrefab = vfxMediumFire;
                    Instantiate(smallSmoke, FirePosition, Quaternion.Euler(-90, 0, 0));
                    //scaleChange = new Vector3(0.7f, 0.7f, 0.7f);
                    break;
                case 4:
                    flamePrefab = largeFlames;
                    //flamePrefab = vfxBigFire;
                    // render the white smoke
                    Instantiate(largeSmoke, FirePosition, Quaternion.Euler(0, 0, 0));
                    Instantiate(whiteSmoke, FirePosition, Quaternion.Euler(0, 0, 0));

                    //scaleChange = new Vector3(0.7f, 0.7f, 0.7f);
                    break;
                default:
                    //Debug.Log("none");
                    initializeObj = false;
                    break;
            }

            if (initializeObj)
            {
                GameObject instance = Instantiate(flamePrefab, FirePosition, Quaternion.Euler(0, 0, 0));
            }
        }
    }

    // Updates grid state for floor 3, considering new fire spread and risk levels.
    void UpdateGridStateForFloor3(int[,] gridState, int[,] riskGrid)
    {
        int[,] newState = new int[gridWidth, gridHeight];
        List<Vector2Int> newFiresThisUpdate = new List<Vector2Int>(); 

        // copy variable for local variables
        for (int x = 0; x < gridWidth; x++)
        {
            for (int y = 0; y < gridHeight; y++)
            {
                newState[x, y] = gridState3[x, y];
            }
        }

        // iterate within the fire building
        for (int x = sceneMinX; x <= sceneMaxX; x++)
        {
            for (int y = sceneMinY; y <= sceneMaxY; y++)
            {
                if (gridState3[x, y] == 1) 
                {
                    // move in 4 directions
                    List<Vector2Int> directions = new List<Vector2Int>
                    {
                        new Vector2Int(0, 1), 
                        new Vector2Int(0, -1), 
                        new Vector2Int(1, 0), 
                        new Vector2Int(-1, 0) 
                    };

                    //find the highest risk
                    int maxRisk = 0; 
                    Vector2Int maxRiskCell = new Vector2Int(-1, -1); 

                    // move in 4 directions
                    foreach (var dir in directions)
                    {
                        int newX = x + dir.x;
                        int newY = y + dir.y;

                        if (newX >= sceneMinX && newX <= sceneMaxX && newY >= sceneMinY && newY <= sceneMaxY && gridState3[newX, newY] == 0)
                        {
                            if (riskGrid[newX, newY] > maxRisk)
                            {
                                maxRisk = riskGrid[newX, newY];
                                maxRiskCell = new Vector2Int(newX, newY);
                            }
                        }
                    }

                    //find the highest risk
                    if (maxRiskCell.x != -1 && maxRiskCell.y != -1)
                    {
                        float chance = Random.Range(0.0f, 1.0f);
                        if (chance < 0.15f*currentRiskMultifier)
                        {
                            newState[maxRiskCell.x, maxRiskCell.y] = 1; 
                            newFiresThisUpdate.Add(maxRiskCell); 
                        }
                    }
                }
            }
        }
        // update state
        gridState3 = newState;
        // render fire
        RenderNewFires3(newFiresThisUpdate);
    }

// Renders new fire instances on the third floor.
    void RenderNewFires3(List<Vector2Int> newFires)
    {
        foreach (var fire in newFires)
        {
            Vector3 FirePosition = new Vector3(fire.x, CalculateHeightForFloor(fire), fire.y);
            Vector3 SmokePosition = new Vector3(fire.x, CalculateHeightForFloor(fire) + 2.5f, fire.y);
            GameObject flamePrefab = null;
            Vector3 scaleChange = Vector3.one; 
            bool initializeObj = true;

            // choose diverse fire effect
            switch (riskGrid3[fire.x, fire.y])
            {
                case 2:
                    flamePrefab = tinyFlames;
                    //flamePrefab = vfxSmallFire;
                    //scaleChange = new Vector3(1.0f, 1.0f, 1.0f); 
                    break;
                case 3:
                    flamePrefab = mediumFlames;
                    //flamePrefab = vfxMediumFire;
                    Instantiate(smallSmoke, FirePosition, Quaternion.Euler(-90, 0, 0));
                    //scaleChange = new Vector3(0.7f, 0.7f, 0.7f);
                    break;
                case 4:
                    flamePrefab = largeFlames;
                    //flamePrefab = vfxBigFire;
                    // render the white smoke
                    Instantiate(largeSmoke, FirePosition, Quaternion.Euler(0, 0, 0));
                    Instantiate(whiteSmoke, FirePosition, Quaternion.Euler(0, 0, 0));

                    break;
                default:
                    //Debug.Log("none");
                    initializeObj = false;
                    break;
            }

            if (initializeObj)
            {
                GameObject instance = Instantiate(flamePrefab, FirePosition, Quaternion.Euler(0, 0, 0));
            }
        }
    }

    // Calculates the appropriate height for fire instances based on the current floor.
    float CalculateHeightForFloor(Vector2Int fire)
    {
        float calculatedHeight = 0;
        if (currentFloor == 1)
        {
            return calculatedHeight; 
        }
        else if (currentFloor == 2)
        {
            if (fire.x == 52 && fire.y == 52)
            {
                return calculatedHeight + 0.96f; 
            }

            if (fire.x == 52 && (fire.y == 53 || fire.y == 54))
            {
                return calculatedHeight + 2.8f; 
            }

            if ((fire.x == 54 || fire.x == 53) && (fire.y >= 52 && fire.y <= 54)) //51到54其实都可以
            {
                return calculatedHeight + 1.72f; 
            }

            return calculatedHeight + 3.39f; 
        }
        else if (currentFloor == 3)
        {
            if (fire.x == 52 && fire.y == 52)
            {
                return calculatedHeight + 3.9f; 
            }

            if ((fire.x == 54 || fire.x == 53) && (fire.y >= 52 && fire.y <= 54)) //51到54其实都可以
            {
                return calculatedHeight + 4.75f; 
            }

            if (fire.x == 52 && (fire.y == 53 || fire.y == 54))
            {
                return calculatedHeight + 5.62f; 
            }

            return calculatedHeight + 6.38f; 
        }
        return 0;
    }


// A* Pathfinding
    // represent the wall in the scene
    const int WALL = 5;
    
    // add walls in the scene
    void AddWallsToGrid(int[,] riskGrid, List<Vector2Int> walls) {
        foreach (var wall in walls) {
            riskGrid[wall.x, wall.y] = WALL;
        }
    }
    
    // Retraces the path from the end node to the start node.
    List<Vector2Int> CollectWalls(int currentRiskFloor)
    {
        List<Vector2Int> walls = new List<Vector2Int>();

        // Common walls for all floors
        for (int y = 52; y <= 54; y++)
        {
            //wall (54,52-54)
            walls.Add(new Vector2Int(54, y));
            //wall (52,52-54)
            walls.Add(new Vector2Int(52, y));
            //wall (52-52,54)
            walls.Add(new Vector2Int(y, 54));
            //wall (52-54,52)
            walls.Add(new Vector2Int(y,52));
        }

        // Floor-specific walls
        if (currentRiskFloor == 3)
        {
            for (int y = 50; y <= 55; y++)
            {
                walls.Add(new Vector2Int(55, y));
            }
        }
        else if (currentRiskFloor == 2)
        {
            for (int y = 55; y <= 57; y++)
            {
                walls.Add(new Vector2Int(57, y));
            }
            for (int y = 49; y <= 51; y++)
            {
                walls.Add(new Vector2Int(57, y));
            }
        }
        else if (currentRiskFloor == 1)
        {
            // wall in 1F stair
            for (int y = 51; y <= 59; y++)
            {
                walls.Add(new Vector2Int(58, y));
                walls.Add(new Vector2Int(57, y));
            }
            //front right side
            for(int y=49; y<=58;y++)
            {
                //walls.Add(new Vector2Int(y,57));
                walls.Add(new Vector2Int(y,58));
            }
            //front left side
            for(int y=60; y<=66;y++)
            {
                //walls.Add(new Vector2Int(y,57));
                walls.Add(new Vector2Int(y,58));
            }
            // rear side
            for(int y=48; y<=70;y++)
            {
                walls.Add(new Vector2Int(y,48));
            }
            //left side
            for (int y = 48; y <= 59; y++)
            {
                walls.Add(new Vector2Int(67,y));
            }
        }
        return walls;
    }
    
    public float CalculateFireRiskLevel(int windValue, int tempValue, int FSIValue)
    {
        // Define thresholds for moderate and high risks based on wind and temperature
        int moderateRiskWindThreshold = 3;
        int highRiskWindThreshold = 5; 
        int moderateRiskTempThreshold = 35; 
        int highRiskTempThreshold = 50;

        float riskLevel = 1;

        // Define multipliers for the FSI value
        int FSIValueMultiplier = FSIValue * 10;

        // Calculate a composite fire risk score
        int fireRiskScore = windValue + tempValue/2 + FSIValueMultiplier;

        // Determine the risk level based on the composite score and individual thresholds
        if (windValue >= highRiskWindThreshold || tempValue >= highRiskTempThreshold || FSIValue == 3)
        {
            if (fireRiskScore>58) { return riskLevel=1.8f; }
            return riskLevel=1.4f;
        }
        else if (windValue >= moderateRiskWindThreshold || tempValue >= moderateRiskTempThreshold || FSIValue >= 1)
        {
            return riskLevel=1f;
        }
        else
        {
            return riskLevel=0.3f;
        }
    }
    
    // A* algorithm for pathfinding, incorporating walls and risk levels in the calculations.
    public static List<Vector2Int> FindPath(int[,] riskGrid,int[,]gridState, Vector2Int start, Vector2Int end)
    {
        int width = riskGrid.GetLength(0);
        int height = riskGrid.GetLength(1);
        
        Dictionary<Vector2Int, PathNode> openSet = new Dictionary<Vector2Int, PathNode>();
        HashSet<Vector2Int> closedSet = new HashSet<Vector2Int>();

        PathNode startNode = new PathNode(start, null, 0, CalculateDistance(start, end));
        openSet.Add(start, startNode);

        while (openSet.Count > 0)
        {
            PathNode currentNode = GetLowestFCostNode(openSet);
            if (currentNode.Position == end)
            {
                // Found the path
                return RetracePath(currentNode);
            }

            openSet.Remove(currentNode.Position);
            closedSet.Add(currentNode.Position);

            foreach (var neighbourPosition in GetNeighbourPositions(currentNode.Position, width, height))
            {
                if (closedSet.Contains(neighbourPosition)|| riskGrid[neighbourPosition.x, neighbourPosition.y] == WALL) continue;
                // calculate the movement cost
                //Debug.Log(
                    //$"X: {neighbourPosition.x}, Y: {neighbourPosition.y}, Risk Level: {riskGrid[neighbourPosition.x, neighbourPosition.y]}");
                int riskCost = riskGrid[neighbourPosition.x, neighbourPosition.y] * 10;
                if (gridState[neighbourPosition.x,neighbourPosition.y]==1)
                {
                    riskCost=riskCost * 2;
                }
                //Debug.Log(riskGrid[neighbourPosition.x, neighbourPosition.y]);
                if (riskGrid[neighbourPosition.x, neighbourPosition.y] == 4)
                {
                    riskCost += 100; 
                }

                int tentativeGCost = currentNode.GCost + 10 + riskCost; // Moving cost is 10 plus the risk level
                if (tentativeGCost < currentNode.GCost || !openSet.ContainsKey(neighbourPosition))
                {
                    PathNode neighbourNode = new PathNode(neighbourPosition, currentNode, tentativeGCost,
                        CalculateDistance(neighbourPosition, end));

                    if (!openSet.ContainsKey(neighbourPosition))
                    {
                        openSet.Add(neighbourPosition, neighbourNode);
                    }
                    else
                    {
                        openSet[neighbourPosition] = neighbourNode;
                    }
                }
            }
        }
        return new List<Vector2Int>(); // Path not found
    }

    // Retraces the path from the end node to the start node.
    private static List<Vector2Int> RetracePath(PathNode endNode)
    {
        List<Vector2Int> path = new List<Vector2Int>();
        PathNode currentNode = endNode;

        while (currentNode != null)
        {
            path.Add(currentNode.Position);
            currentNode = currentNode.PreviousNode;
        }

        path.Reverse();
        return path;
    }

    // Gets the node with the lowest F cost from the open set.
    private static PathNode GetLowestFCostNode(Dictionary<Vector2Int, PathNode> openSet)
    {
        PathNode lowestFCostNode = null;
        foreach (var node in openSet.Values)
        {
            if (lowestFCostNode == null || node.FCost < lowestFCostNode.FCost)
            {
                lowestFCostNode = node;
            }
        }
        return lowestFCostNode;
    }

    // Gets neighboring positions for a given grid position.
    private static List<Vector2Int> GetNeighbourPositions(Vector2Int position, int width, int height)
    {
        List<Vector2Int> positions = new List<Vector2Int>();

        if (position.x > 49) positions.Add(new Vector2Int(position.x - 1, position.y));
        if (position.x < 67) positions.Add(new Vector2Int(position.x + 1, position.y));
        if (position.y > 49) positions.Add(new Vector2Int(position.x, position.y - 1));
        if (position.y < 57) positions.Add(new Vector2Int(position.x, position.y + 1));

        return positions;
    }

    // Calculates the distance between two points on the grid.
    private static int CalculateDistance(Vector2Int a, Vector2Int b)
    {
        return Mathf.Abs(a.x - b.x) + Mathf.Abs(a.y - b.y);
    }
    
    // class for pathfinding
    public class PathNode
    {
        public Vector2Int Position;
        public PathNode PreviousNode;
        public int GCost;
        public int HCost;

        public int FCost
        {
            get { return GCost + HCost; }
        }

        public PathNode(Vector2Int position, PathNode previousNode, int gCost, int hCost)
        {
            this.Position = position;
            this.PreviousNode = previousNode;
            this.GCost = gCost;
            this.HCost = hCost;
        }
    }

    // Finds and displays the safest evacuation path based on the risk grid.
    public void FindAndDisplayPath(int[,] riskGrid,int[,]gridState, Vector2Int startPoint, Vector2Int endPoint)
    {
        int width = riskGrid.GetLength(0);
        int height = riskGrid.GetLength(1);
        int[,] clonedRiskGrid = new int[width, height];
        
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                clonedRiskGrid[x, y] = riskGrid[x, y];
            }
        }
        
        AddWallsToGrid(clonedRiskGrid,CollectWalls(currentRiskFloor));
        List<Vector2Int> path = FindPath(clonedRiskGrid, gridState,startPoint, endPoint);

        if (path != null && path.Count > 0)
        {
            DisplayPath(path);
            //Debug.Log("path found.");
        }
        else
        {
            Debug.Log("No path found.");
        }
    }

    // Displays the evacuation path in the game world.
    private void DisplayPath(List<Vector2Int> path)
    {
        float pathHeight = .0f;
        switch (currentRiskFloor)
        {
            case 1:
                pathHeight = 0.21f;
                break;
            case 2:
                pathHeight = 3.19f;
                break;
            case 3:
                pathHeight = 6.21f;
                break;
        }
        StringBuilder pathString = new StringBuilder();
        pathString.Append("Path: ");

        foreach (Vector2Int point in path)
        {
            pathString.AppendFormat("({0}, {1}) ", point.x, point.y);
        }
        
        foreach (Vector2Int point in path)
        {
            if (point.x >= 52 && point.x <= 54 && point.y <= 54 && point.y >= 52)
            {
                continue;
            }
            // initialize the prefab of path mark
            Vector3 worldPosition = new Vector3(point.x, pathHeight, point.y); 
            Instantiate(pathPrefab, worldPosition, Quaternion.Euler(0, 0, 0));
        }
    }

    // Cleans up old path markers before recalculating the path to reflect the changing environment.
    public void DeleteSign()
    {
        GameObject[] objectsToDestroy = GameObject.FindGameObjectsWithTag("GuideSign");
        foreach(GameObject obj in objectsToDestroy)
        {
            Destroy(obj);
        }
    }
}