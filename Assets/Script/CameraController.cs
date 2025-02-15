using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    private float mouseX, mouseY;
    public float mouseSensitivity;
    public Transform firefighter;
    public float xRotation;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        mouseX=Input.GetAxis ("Mouse X")*mouseSensitivity*Time.deltaTime;
        mouseY=Input.GetAxis ("Mouse Y")*mouseSensitivity*Time.deltaTime;
        firefighter.Rotate(Vector3.up*mouseX);
        xRotation-=mouseY;
        transform.localRotation=Quaternion.Euler(xRotation,0,0);
    }
}
