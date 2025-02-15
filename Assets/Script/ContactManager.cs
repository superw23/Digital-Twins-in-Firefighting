using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ContactManager : MonoBehaviour
{
    [SerializeField] private FireController fireController;
    private void OnTriggerEnter(Collider other)
    {
        switch(other.gameObject.tag) {
            case "TinyFlames":
                fireController.IncreaseContactWithTinyFlame();
                break;
            case "MediumFlames":
                fireController.IncreaseContactWithMediumFlame();
                break;
            case "LargeFlames":
                fireController.IncreaseContactWithLargeFlame();
                break;
        }
    }
}
