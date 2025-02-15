using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class ConnectionTester : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        StartCoroutine(TestConnection());
    }

    IEnumerator TestConnection()
    {
        string url = "http://localhost:5000/testConnection";
        UnityWebRequest www = UnityWebRequest.Get(url);
        yield return www.SendWebRequest();

        if (www.isNetworkError || www.isHttpError)
        {
            Debug.LogError("Error: " + www.error);
            Debug.LogError("please connect Flask Server");
        }
        else
        {
            Debug.Log("Received: " + www.downloadHandler.text);
        }
    }
}
